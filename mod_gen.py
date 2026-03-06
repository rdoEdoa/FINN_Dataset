import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.export import export_qonnx
import os
import random
import warnings
import sys

# Silence the specific PyTorch Named Tensor warning
warnings.filterwarnings("ignore", message=".*Named tensors.*")

OUTPUT_DIR = "dataset"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, "weights")
ONNX_DIR = os.path.join(OUTPUT_DIR, "onnx_models")
NUM_SMALL = 2
NUM_LARGE = 0
IMG_H = 32
IMG_W = 32

def main():

   # Create the output directories if they don't exist
   if not os.path.exists(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)
   if not os.path.exists(MODELS_DIR):
      os.makedirs(MODELS_DIR)
   if not os.path.exists(WEIGHTS_DIR):
      os.makedirs(WEIGHTS_DIR)
   if not os.path.exists(ONNX_DIR):
      os.makedirs(ONNX_DIR)

   # Generate small models
   for i in range(NUM_SMALL):
      print(f"Generating small model {i+1}/{NUM_SMALL}...")
      
      max_att = 5
      isValid = False
      
      for attempt in range(max_att): # Max attempts in order to avoid infinite loops
         in_ch = random.choice([1, 3, 10]) 
         dummy_input = torch.randn(1, in_ch, IMG_H, IMG_W)
         
         try:
            model = RandomBlock(mode='small', in_ch=in_ch, img_h=IMG_H, img_w=IMG_W)
            _ = model(dummy_input) # Test if the full forward pass works
            isValid = True
            break
         except RuntimeError as e:
            # Catch errors in the model
            print(f"[Attempt {attempt+1}/{max_att}] Model generation failed with error: {e}")
            continue
      
      if not isValid:
         print(f"Failed to generate a valid model after {max_att} attempts. Skipping small model {i:04d}.")
         continue      
      
      # Export the model to .py format
      model_path = os.path.join(MODELS_DIR, f"small_model_{i:04d}.py")
      with open(model_path, 'w') as f:
         f.write("import torch\n")
         f.write("import torch.nn as nn\n")
         f.write("import brevitas.nn as qnn\n\n")
         
         f.write("class GeneratedModel(nn.Module):\n")
         f.write("   def __init__(self):\n")
         f.write("      super(GeneratedModel, self).__init__()\n")
         
         # Write the layer definitions
         for line in model.init_code_lines:
             f.write(f"      {line}\n")
             
         f.write("\n   def forward(self, x):\n")
         
         # Write the forward pass
         for line in model.forward_code_lines:
             f.write(f"      {line}\n")
             
         f.write("      return x\n")
         
      # Save the weights, needed for future steps
      weights_path = os.path.join(WEIGHTS_DIR, f"small_model_{i:04d}.pth")
      torch.save(model.state_dict(), weights_path)
      
      # Export the model to QONNX format
      onnx_path = os.path.join(ONNX_DIR, f"small_model_{i:04d}.onnx")
      export_qonnx(model, dummy_input, onnx_path)

class RandomBlock(nn.Module):
   # img_h and img_w are passed to calculate the flatten size
   def __init__(self, mode='small', in_ch=3, img_h=32, img_w=32):
      super(RandomBlock, self).__init__()
      
      self.init_code_lines = []
      self.forward_code_lines = []
      
      if mode == 'small':
         target_layers = random.randint(1, 3)
         possible_ch = [32, 64, 128]
         stride_prob = 0.4
      else: 
         target_layers = random.randint(10, 25)
         possible_ch = [64, 128, 256, 512]
         stride_prob = 0.2
         
      bw = random.choice([2, 4, 8])
      
      # Needed layer to format the inputs
      self.quant_inp = qnn.QuantIdentity(bit_width=bw, return_quant_tensor=True)
      self.init_code_lines.append(f"self.quant_inp = qnn.QuantIdentity(bit_width={bw}, return_quant_tensor=True)")
      self.forward_code_lines.append("x = self.quant_inp(x)")
      
      current_ch = in_ch
      self.layer_names = []
      
      for i in range(target_layers):
         if i == 0:
            current_ch = in_ch
         else:
            current_ch = out_ch
            
         # Parameters randomization
         out_ch = random.choice(possible_ch)
         stride = 2 if random.random() < stride_prob else 1
         kernel = random.choice([3, 5])
         pad = random.choice([0, 1])
         
         # Conv layer
         conv_layer_name = 'QuantConv2d'
         conv_class = getattr(qnn, conv_layer_name)
         
         conv = conv_class(
            in_channels = current_ch,
            out_channels = out_ch,
            kernel_size = kernel,
            stride = stride,
            padding = pad,
            bias = False,
            weight_bit_width = bw           
         )
         
         # Activation layer
         act = qnn.QuantReLU(bit_width=bw)
         
         # Pooling layer, only half the time
         if random.random() < 0.5:
            isPool = True
            pool = nn.MaxPool2d(kernel_size=random.choice([2, 3]), stride=random.choice([1, 2]))
         else:
            isPool = False
           
         conv_name = f"conv_{i}"
         act_name = f"act_{i}"
         
         setattr(self, conv_name, conv)
         setattr(self, act_name, act)
         self.layer_names.append(conv_name)
         self.layer_names.append(act_name)
         
         if isPool:
            pool_name = f"pool_{i}"
            setattr(self, pool_name, pool)
            self.layer_names.append(pool_name)
         
         self.init_code_lines.append(
             f"self.{conv_name} = qnn.{conv_layer_name}("
             f"in_channels={current_ch}, out_channels={out_ch}, "
             f"kernel_size={kernel}, stride={stride}, padding={pad}, "
             f"bias=False, weight_bit_width={bw})"
         )
         self.init_code_lines.append(f"self.{act_name} = qnn.QuantReLU(bit_width={bw})")
         if isPool:
             self.init_code_lines.append(
                 f"self.{pool_name} = nn.MaxPool2d("
                 f"kernel_size={pool.kernel_size}, stride={pool.stride})"
             )
         
         self.forward_code_lines.append(f"x = self.{conv_name}(x)")
         self.forward_code_lines.append(f"x = self.{act_name}(x)")
         if isPool:
             self.forward_code_lines.append(f"x = self.{pool_name}(x)")
             
      # Flatten the output before the final fully connected layers
      self.flatten = nn.Flatten()
      setattr(self, "flatten", self.flatten)
      self.layer_names.append("flatten")
      self.init_code_lines.append("self.flatten = nn.Flatten()")
      self.forward_code_lines.append("x = self.flatten(x)")
      
      # To solve problems with the unknown flatten size
      # 1. Create a dummy image tensor
      dummy_x = torch.zeros(1, in_ch, img_h, img_w)
      
      # 2. Push it through the input quantizer
      dummy_x = self.quant_inp(dummy_x)
      
      # 3. Push it through all the previous layers
      for name in self.layer_names:
         layer = getattr(self, name)
         dummy_x = layer(dummy_x)
         
      # 4. Read the exact output size
      current_in_features = dummy_x.shape[1]
      
      # Final layers
      for i in range(random.randint(1, 2)):
         out_feat = random.choice(possible_ch)
         
         quant = qnn.QuantLinear(in_features=current_in_features, out_features=out_feat, weight_bit_width=bw)
         setattr(self, f"quant_{i}", quant)
         self.layer_names.append(f"quant_{i}")
         
         self.init_code_lines.append(
             f"self.quant_{i} = qnn.QuantLinear(in_features={current_in_features}, out_features={out_feat}, weight_bit_width={bw})"
         )
         self.forward_code_lines.append(f"x = self.quant_{i}(x)") 
         
         fc_act = qnn.QuantReLU(bit_width=bw)
         setattr(self, f"fc_act_{i}", fc_act)
         self.layer_names.append(f"fc_act_{i}")
         
         self.init_code_lines.append(f"self.fc_act_{i} = qnn.QuantReLU(bit_width={bw})")
         self.forward_code_lines.append(f"x = self.fc_act_{i}(x)")
         
         # Update the input features for the next Linear layer
         current_in_features = out_feat
         
   def forward(self, x):
      x = self.quant_inp(x)
      for name in self.layer_names:
         layer = getattr(self, name)
         x = layer(x)
      return x

if __name__ == "__main__":
   main()
   print("Models generation complete.")