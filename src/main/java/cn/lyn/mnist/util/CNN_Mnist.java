package cn.lyn.mnist.util;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CNN_Mnist {
    private static Logger log = LoggerFactory.getLogger(CNN_Mnist.class);
    public static int num_progress=0;//全局变量，用来更新前端的进度条的

    public static String tranModel() throws IOException {
        int nChannels = 1; //single channel for grayscale images单通道灰度图
        int outputNum = 10; // The number of possible outcomes潜在结果的数量（比如0-9共10个）
        int batchSize = 64; // Test batch size每一步抓取的样例数量
        int nEpochs = 10; // Number of training epochs将给定数据集处理的周期数
        int iterations = 1; // Number of training iterations训练迭代次数
        int seed = 123; //随机数种子，用来确保训练时使用的初始权重维持一致
 
//        URL tranUrl = CNN_Mnist.class.getClassLoader().getResource("mnist/mnist_png/training");
//        URL testUrl = CNN_Mnist.class.getClassLoader().getResource("mnist/mnist_png/testing");
//        File trainData = new File(tranUrl.getFile());//加载训练图集
//        File testData = new File(testUrl.getFile());//加载测试图集
//      
//        
//
//
//
//        //文件输入拆分，将根目录拆分成文件
//        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
//        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
//
//        //用指定宽高、通道加载图片文件
//        ImageRecordReader recordReader = new ImageRecordReader(28, 28, 1, new ParentPathLabelGenerator());
//        recordReader.initialize(train);
//
//        //数据集迭代器：处理数据集并准备神经网络的数据
//        DataSetIterator mnistTrain = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
//        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);//数据规范化到0-1之间
//
//        scaler.fit(mnistTrain);//迭代数据集，使其规范化
//        mnistTrain.setPreProcessor(scaler);
//       
        //Get the DataSetIterators:使用封装好的mnist数据集，这样比用上面加载并处理原始图片的方法快很多
      //batchSize表示批处理大小，true表示是训练集,false表示是测试集，seed表示随机数种子用于打乱集合
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);


        //神经网络配置类，通过这个配置类设置各种超参数来搭建神经网络
        MultiLayerConfiguration conf = new NeuralNetConfiguration
                .Builder()
                .seed(seed)    //将一组随机生成的数作为初始权重，用于运行期间的重现性

                .activation(Activation.RELU)//设置默认激活函数，如果某一层没有指定激活函数，则使用默认的
                .weightInit(WeightInit.XAVIER)//初始化权重方案
                .updater(new Nesterovs(0.0015, 0.98))//权重更新方案，第一个参数是学习速率，第二个参数是动量
                .l2(0.0005) // regularize learning model使用l2正则化，防止个别权重对整体结果产生过大影响
                .list()//指定网络中层的数量；它会将您的配置复制n次，建立分层的网络结构
                .layer(0, new ConvolutionLayer.Builder(5, 5)//卷积层5*5
                        .nIn(nChannels)//通道数
                        .stride(1, 1)//卷积神经网络进行卷积时的步长
                        .nOut(20)
                        .activation(Activation.IDENTITY).//激活函数
                                build())
                .layer(1, new SubsamplingLayer//子采样层
                        .Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)  //2*2内核
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer  //全连接层
                        .Builder()
                        .activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(5, new OutputLayer  //输出层
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1))//卷积数据的输入类型，高28，宽28，深1
               .backprop(true).pretrain(false)//后向传播，不预先训练
                .build();
        
        //由配置创建神经网络模型并初始化
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(5));//迭代的监听器，设定打印频率为5

        log.info("******EVALUATE MODEL******");

        for (int i = 0; i < nEpochs; i++) {
        	num_progress=i*10;
            model.fit(mnistTrain);//基于给定的数据集迭代器训练模型
            
            log.info("*** Completed epoch {} ***", i);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatures()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("******SAVE TRAINED MODEL******");
        // Details

        // Where to save model
        File locationToSave = new File("trained_mnist_model.zip");
        //File locationToSave = new ClassPathResource("trained_mnist_model.zip").getFile();
        // boolean save Updater
        boolean saveUpdater = true;

        // ModelSerializer needs modelname, saveUpdater, Location

        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        return eval.stats();
        
    }
}
