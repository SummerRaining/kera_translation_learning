�ո�ѧϰ��keras��ʹ��keras��һ���򵥵�inception-v3Ǩ��ѧϰ��
�ṹ���̶�ǰ�����е�������ȡ�㣬�ں������ȫ���Ӳ���Ϊ�б�������ѵ��
ʹ��inception-v3��Ǩ��ѧϰ�����岽�裺
    1������inceptionV3ģ�ͣ�ʹ��base_model = InceptionV3(weights = "imagenet',include_top = False)
    2�����base_model()��ȡ������������x = base_model.output
    3����GlobalAveragePooling2D()(x)��x���ֻ��n_channel��һά����,˼����һ��filter����ͼ���һ��������
         ��output�������ȫ���Ӳ㣬�õ�output,ʹ�ú���ʽ����ģ�͡��õ��µ�ģ��model��
    4��ע��model�ǻ�����ǰ��ģ��base_model�����ĸı�base_model��ʹmodel��֮�����ı䣬
       �ı�base_model�е�����layer��trainable���ԣ�base_model.layers()��������layer���б�
    5��model.compile()�涨ѧϰ���̡�optimizer,loss,metrics
    5������ImageDataGenerator(),ѵ���������Ͳ�������������ʼѵ��model.fit_generator(generator)
ͼ��Ԥ����������ĵ����⣺https://keras-cn.readthedocs.io/en/latest/preprocessing/image/#_1
ͼ���ļ��нṹ��
trainAndval:
    --train:
        --flower1
        --flower2
        ....
    --val:
        --flower1
        --flower2
        ....
���ų����ֽṹ������ʹ��generator = ImageDataGenerator()������������
ʹ��generator.flow_from_directory()������ȡͼ��

�����
����һ��Сʱ��ѵ����ʹ���Կ�1080ti����ѵ�����ϵ�׼ȷ��0.92����֤����0.78��Ч����Ԥ�����ܴ󣬲�֪��Ϊɶ��������취���������