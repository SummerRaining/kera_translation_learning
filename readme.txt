刚刚学习了keras，使用keras做一个简单的inception-v3迁移学习。
结构：固定前面所有的特征提取层，在后面添加全连接层作为判别器进行训练
使用inception-v3做迁移学习，具体步骤：
    1、导入inceptionV3模型，使用base_model = InceptionV3(weights = "imagenet',include_top = False)
    2、获得base_model()提取出来的特征。x = base_model.output
    3、用GlobalAveragePooling2D()(x)将x变成只有n_channel的一维向量,思想是一个filter代表图像的一种特征。
         对output添加两个全连接层，得到output,使用函数式构建模型。得到新的模型model。
    4、注意model是基于以前的模型base_model而来的改变base_model会使model随之发生改变，
       改变base_model中的所有layer的trainable属性，base_model.layers()会获得所有layer的列表
    5、model.compile()规定学习过程。optimizer,loss,metrics
    5、构建ImageDataGenerator(),训练生成器和测试生成器。开始训练model.fit_generator(generator)
图像预处理的中文文档在这：https://keras-cn.readthedocs.io/en/latest/preprocessing/image/#_1
图像文件夹结构：
trainAndval:
    --train:
        --flower1
        --flower2
        ....
    --val:
        --flower1
        --flower2
        ....
安排成这种结构，可以使用generator = ImageDataGenerator()构建生成器后，
使用generator.flow_from_directory()方法读取图像。

结果：
经过一个小时的训练（使用显卡1080ti），训练集上的准确率0.92，验证集上0.78。效果与预期相差很大，不知道为啥？正在想办法解决。。。