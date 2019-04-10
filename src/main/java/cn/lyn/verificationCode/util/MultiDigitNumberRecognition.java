package cn.lyn.verificationCode.util;


import java.io.File;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;




/**
 * @Description This is a demo that multi-digit number recognition. The maximum length is 6 digits.
 * If it is less than 6 digits, then zero is added to last
 * Training set: There were 14108 images, and they were used to train a model.
 * Testing set: in total 108 images,they copied from the training set,mainly to determine whether it's good that the model fited training data
 * Verification set: The total quantity of the images has 248 that's the unknown image,the main judgment that the model is good or bad
 * Other: Using the current architecture and hyperparameters, the accuracy of the best model prediction validation set is (215-227) / 248 with different epochs
 * of course, if you're interested, you can continue to optimize.
 * @author WangFeng
 */
public class MultiDigitNumberRecognition {


    private static final Logger log = LoggerFactory.getLogger(MultiDigitNumberRecognition.class);

    private static long seed = 123;
    private static int epochs = 50;
 
    private static int batchSize = 15;
    private static String rootPath = System.getProperty("user.dir");

    //private static String modelDirPath = rootPath.substring(0, rootPath.lastIndexOf(File.separatorChar)) + File.separatorChar + "out" + File.separatorChar + "models";
    private static String modelDirPath = rootPath + File.separatorChar + "out" + File.separatorChar + "models";
    public static String modelPath = modelDirPath + File.separatorChar + "validateCodeCheckModel.json";
    
    public static int progressNum=0;

    public static void tranningModel() throws Exception {
    	System.out.println("************************");
    	System.out.println(rootPath);
    	System.out.println(modelDirPath);
    	System.out.println(modelPath);
    	System.out.println("************************");
    	
        long startTime = System.currentTimeMillis();
        System.out.println(startTime);

        File modelDir = new File(modelDirPath);

        // create directory
        boolean hasDir = modelDir.exists() || modelDir.mkdirs();
        log.info( modelPath );
        //create model
        ComputationGraph model =  createModel();
        //monitor the model score
//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10));//打印频率为10
        
        //construct the iterator
        MultiDataSetIterator trainMulIterator = new MultiRecordDataSetIterator(batchSize, "train");
        MultiDataSetIterator testMulIterator = new MultiRecordDataSetIterator(batchSize,"test");
        MultiDataSetIterator validateMulIterator = new MultiRecordDataSetIterator(batchSize,"validate");
        //fit
        for ( int i = 0; i < epochs; i ++ ) {
        	progressNum=i*2;
            System.out.println("Epoch=====================" + i);
            model.fit(trainMulIterator);
        }
        //保存模型
        ModelSerializer.writeModel(model, modelPath, true);
     
       // ModelSerializer.writeModel(model, "trained_digit_recognition.json", true);
        long endTime = System.currentTimeMillis();
        System.out.println("=============run time=====================" + (endTime - startTime));

        System.out.println("=====eval model=====test==================");
        modelPredict(model, testMulIterator);

        System.out.println("=====eval model=====validate==================");
        modelPredict(model, validateMulIterator);

    }

    public static ComputationGraph createModel() {

        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)//设置每层默认的梯度归一化算法，为l2正则化算法
            .l2(1e-3)//每层默认的l2正则化算法，防止个别权重对结果产生较大影响
            .updater(new Adam(1e-3))//使用Adam进行梯度更新
            .weightInit( WeightInit.XAVIER_UNIFORM)//设置默认的权重更新方案
            .graphBuilder()//创建一个GraphBuilder (为了创建ComputationGraphConfiguration)
            .addInputs("trainFeatures")//指定输入层的名称
            .setInputTypes(InputType.convolutional(60, 160, 1))//指定输入层的类型
            .setOutputs("out1", "out2", "out3", "out4", "out5", "out6")//指定输出层的名称
            
            //卷积层，三个数组分别表示，卷积内核，步长内核，填充内核
            .addLayer("cnn1",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                .nIn(1).nOut(48).activation( Activation.RELU).build(), "trainFeatures")//Activation.RELU激活函数：y=max(0,x)
            .addLayer("maxpool1",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn1")//子采样层，也被称为卷积神经网络中的池，PoolingType.MAX表示输出是输入值的最大值
            .addLayer("cnn2",  new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(64).activation( Activation.RELU).build(), "maxpool1")
            .addLayer("maxpool2",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,1}, new int[]{2, 1}, new int[]{0, 0})
                .build(), "cnn2")
            .addLayer("cnn3",  new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(128).activation( Activation.RELU).build(), "maxpool2")
            .addLayer("maxpool3",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn3")
            .addLayer("cnn4",  new ConvolutionLayer.Builder(new int[]{4, 4}, new int[]{1, 1}, new int[]{0, 0})
                .nOut(256).activation( Activation.RELU).build(), "maxpool3")
            .addLayer("maxpool4",  new SubsamplingLayer.Builder(PoolingType.MAX, new int[]{2,2}, new int[]{2, 2}, new int[]{0, 0})
                .build(), "cnn4")
            .addLayer("ffn0",  new DenseLayer.Builder().nOut(3072)//全连接前向反馈层
                .build(), "maxpool4")
            .addLayer("ffn1",  new DenseLayer.Builder().nOut(3072)
                .build(), "ffn0")
            .addLayer("out1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)//使用的损失函数
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")//输出层，等于DenseLayer+LossLayer
            .addLayer("out2", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out3", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out4", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out5", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .addLayer("out6", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10).activation(Activation.SOFTMAX).build(), "ffn1")
            .build();

        // Construct and initialize model
        ComputationGraph model = new ComputationGraph(config);
        model.init();

        return model;
    }

    public static void modelPredict(ComputationGraph model, MultiDataSetIterator iterator) {
        int sumCount = 0;
        int correctCount = 0;

        while (iterator.hasNext()) {
            MultiDataSet mds = iterator.next();
            INDArray[]  output = model.output(mds.getFeatures());
            INDArray[] labels = mds.getLabels();
            int dataNum = batchSize > output[0].rows() ? output[0].rows() : batchSize;//如果某一次批处理时，数据的条数小于批处理数，则使用真实的条数来做下面的循环
            for (int dataIndex = 0;  dataIndex < dataNum; dataIndex ++) {
                String reLabel = "";
                String peLabel = "";
                INDArray preOutput = null;
                INDArray realLabel = null;
                for (int digit = 0; digit < 6; digit ++) {//总共有6个数字
                    preOutput = output[digit].getRow(dataIndex);
                    peLabel += Nd4j.argMax(preOutput, 1).getInt(0);//在peLable上拼上预测的结果的可能性最大的值
                    realLabel = labels[digit].getRow(dataIndex);
                    reLabel += Nd4j.argMax(realLabel, 1).getInt(0);//在reLable上拼上标签在该位置的值（实际值）
                }
                if (peLabel.equals(reLabel)) {//比较一张图片完整的预测结果和真实值是否相等
                    correctCount ++;
                }
                sumCount ++;
                log.info("real image {}  prediction {} status {}",  reLabel,peLabel, peLabel.equals(reLabel));
            }
        }
        iterator.reset();
        System.out.println("validate result : sum count =" + sumCount + " correct count=" + correctCount );
    }
}
