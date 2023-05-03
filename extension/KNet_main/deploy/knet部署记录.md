# knet部署记录

## knet简介：
K-Net: Towards Unified Image Segmentation   
code: https://github.com/ZwwWayne/K-Net/    
paper: https://arxiv.org/abs/2106.14855 

## pytorch2torchscript  
使用torchscript原因：    
（1）转换友好。相同版本中，torchscript算子与pyotrch算子一一对应，避免某些算子在转换时不存在对应实现算子的问题，且推断结果完全一致，因此转换比较友好。    
（题外话，虽然pytorch提供onnx export，这个过程会有各种问题，该接口转onnx本质其实也是先转torchscript，当然这是可解决的，摸坑...）       
（2）项目已有环境，不需要更改。项目opencv版本为4.1，无法使用相关主流框架模型文件推断。如果是opencv4.5及以上，可支持onnx、openvino、darknet等...    
（3）项目对速度要求相对没那么大。   

### 环境配置
```text
cpp:    
libtorch==1.6.0(cpu version)

python:
torch==1.6.0+cu101
torchvision==0.7.0+cu101
mmdet==2.25.0
mmcv-full==1.5.1
```

### 转换思路
（1）目前推断图像尺寸是固定的，因此不考虑动态尺寸的情况。   
（2）尽量减少重构的工作量。      
（3）`jit.trace`和`jit script`混合转换。各子模块分别转换，然后串在一起。    
backbone、neck、head部分使用`jit.trace`，因为这些推断流大多数是固定的，少部分存在分支（如head的某些部分）则使用`torch.jit._script_if_tracing`规避。    
decoder部分使用`jit.script`，因为输入不确定，同时需要重构部分函数，使其符合转换语法。    
（4）重构不合理的部分。    
有些问题可能在所有版本均会出现。        
如`jit.trace`转换某些算子时会产生静态变量（尤其是张量的设备类型变量），使得转换后模型需要与转换时的输入一致，限制模型的推断方式。    
有些问题可能在不同版本中出现。     
如torch 1.8.x版本的`x=torch.cat([x[-2:], x, x[2:]], dim=0)`报出内存错误，无法转换。 

### 转换问题

分支类运算：
```python
nonzero_inds = sigmoid_masks > 0.5
-->
@torch.jit._script_if_tracing
def get_mask(pred):
    return pred > 0.5
nonzero_inds = get_mask(sigmoid_masks)
```

重构模块：   
(1)ConvKernelHeadInference:     
将`ConvKernelHead`推断部分抽出并进行重构，命名为`ConvKernelHeadInference`。      
以`ConvKernelHead`为输入，将需要的属性给予`ConvKernelHeadInference`，forward部分拷贝源码后修改重构。      
(2)KernelUpdateHeadInference:   
基本同`ConvKernelHeadInference`，去除forward函数部分不需要的输入参数和代码块。
(3)KernelIterHeadInference:     
基本同`ConvKernelHeadInference`

规则：     
(1)     
(1)返回值不允许出现None         
(2)     


