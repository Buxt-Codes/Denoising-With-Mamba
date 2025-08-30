from torch.utils.tensorboard import SummaryWriter
import torch
import time

writer = SummaryWriter('runs/example_experiment')

for step in range(100):
    writer.add_scalar('loss', torch.rand(1).item(), step)
    writer.add_scalar('accuracy', torch.rand(1).item(), step)

writer.close()

# Launch TensorBoard
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', r'C:\Users\ian\Desktop\Coding\ReviewClassification\model\weights\cross_attn\CrossAttentionTransformer_20250828_203332\writer'])
url = tb.launch()
print(f"TensorBoard running at {url}")
print("TensorBoard running at http://localhost:6006/")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
writer.close()