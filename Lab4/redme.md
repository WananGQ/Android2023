# 实现基本的图像分类APP
## 下载初始代码

```bash
git clone https://github.com/hoitab/TFLClassify.git
```
## 运行代码
![手机运行]()
## 向应用中添加TensorFlow Lite
![添加lite]()
## 检查代码中的TODO项
TODO1

```bash
// TODO 1: Add class variable TensorFlow Lite Model
private val flowerModel = FlowerModel.newInstance(ctx)
```

TODO2

```bash
// TODO 2: Convert Image to Bitmap then to TensorImage
            val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy))
```

TODO3

```bash
 val outputs = flowerModel.process(tfImage)
                .probabilityAsCategoryList.apply {
                    sortByDescending { it.score } // Sort with highest confidence first
                }.take(MAX_RESULT_DISPLAY) // take the top results
```

TODO4

```bash
// TODO 4: Converting the top probability items into a list of recognitions
for (output in outputs) {
                items.add(Recognition(output.label, output.score))
            }
```

TODO6

```bash
// TODO 6. Optional GPU acceleration
val compatList = CompatibilityList()

        val options = if(compatList.isDelegateSupportedOnThisDevice) {
            Log.d(TAG, "This device is GPU Compatible ")
            Model.Options.Builder().setDevice(Model.Device.GPU).build()
        } else {
            Log.d(TAG, "This device is GPU Incompatible ")
            Model.Options.Builder().setNumThreads(4).build()
        }
```
## 对图像进行处理并生成结果，主要包含下述操作：
按照属性score对识别结果按照概率从高到低排序
列出最高k种可能的结果，k的结果由常量MAX_RESULT_DISPLAY定义

```bash
override fun analyze(imageProxy: ImageProxy) {
  ...
  // TODO 3: Process the image using the trained model, sort and pick out the top results
  val outputs = flowerModel.process(tfImage)
      .probabilityAsCategoryList.apply {
          sortByDescending { it.score } // Sort with highest confidence first
      }.take(MAX_RESULT_DISPLAY) // take the top results

  ...
}

```
将识别的结果加入数据对象Recognition 中，包含label和score两个元素。后续将用于RecyclerView的数据显示
```bash
override fun analyze(imageProxy: ImageProxy) {
  ...
  // TODO 4: Converting the top probability items into a list of recognitions
  for (output in outputs) {
      items.add(Recognition(output.label, output.score))
  }
  ...
}

```
将原先用于虚拟显示识别结果的代码注释掉或者删除`

```bash
// START - Placeholder code at the start of the codelab. Comment this block of code out.
for (i in 0..MAX_RESULT_DISPLAY-1){
    items.add(Recognition("Fake label $i", Random.nextFloat()))
}
// END - Placeholder code at the start of the codelab. Comment this block of code out.

```
## 最后运行
![运行]()
