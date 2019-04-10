package cn.lyn.verificationCode.util;

import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;


/**
 * @author WangFeng
 */
public class MulRecordDataLoader extends NativeImageLoader implements Serializable {
    private static final Logger log = LoggerFactory.getLogger(MulRecordDataLoader.class);

    private static int height = 60;//图片高度
    private static int width = 160;//图片宽度
    private static int channels = 1;//图片通道，默认是单通道灰度图
    private File fullDir = null;//图片所在的根文件夹
    private Iterator<File> fileIterator;//文件迭代器
    private int numExample = 0;//文件数量

    private  List<String> labels = new ArrayList<>();//记录下图片的标签
    public  List<String> getLabels(){
    	if(labels.size()!=0) {Collections.shuffle(labels);return labels;}
    	int num=0;
    	 while (fileIterator.hasNext()&&num<1000) {
             File image = fileIterator.next();//从文件迭代器中取出下一个图片文件
             String imageName = image.getName().substring(0,image.getName().lastIndexOf('.'));//获取图片文件的名字，就是这个图片验证码的真实值        
             labels.add(imageName);//将图片集中的图片标签记录下来，以备前端页面使用
             num++;
    	 }
    	return labels;
    }
    public MulRecordDataLoader() {}
    public MulRecordDataLoader(String dataSetType) {
        this( height, width, channels, null, dataSetType);
    }
    public MulRecordDataLoader(ImageTransform imageTransform, String dataSetType)  {
        this( height, width, channels, imageTransform, dataSetType );
    }
    public MulRecordDataLoader(int height, int width, int channels, ImageTransform imageTransform, String dataSetType) {
        super(height, width, channels, imageTransform);
        this.height = height;
        this.width = width;
        this.channels = channels;
        try {//加载图片所在根文件夹
        	//这种方式在打包成jar后会报错，因为这种方式访问不到jar包内的资源
            this.fullDir = fullDir != null && fullDir.exists() ? fullDir : new ClassPathResource("/static/captchaImage").getFile();
//            this.fullDir = fullDir != null && fullDir.exists() ? fullDir : new File(this.getClass().getResource("/captchaImage").getFile());    
//        	String rootPath = System.getProperty("user.dir");
//            this.fullDir = fullDir != null && fullDir.exists() ? fullDir : new File(rootPath.substring(0, rootPath.lastIndexOf(File.separator))+File.separator+"captchaImage");    
        } catch (Exception e) {
           // log.error("the datasets directory failed, plz checking", e);
            throw new RuntimeException( e );
        }
        this.fullDir = new File(fullDir, dataSetType);//根据dataSetType的值确定使用的是哪一个子文件夹
        load();
    }

    protected void load() {
        try {
            List<File> dataFiles = (List<File>) FileUtils.listFiles(fullDir, new String[]{"jpeg"}, true );//列出当前文件夹下的所有JPEG类型的文件
            Collections.shuffle(dataFiles);//使用默认随机源打乱文件列表
            fileIterator = dataFiles.iterator();//返回文件按列表的迭代器
            numExample = dataFiles.size();//返回文件列表中文件的数量
  
        } catch (Exception var4) {
            throw new RuntimeException( var4 );
        }
    }

    public MultiDataSet convertDataSet(int num) throws Exception {//将图片转换成数据集对象，一次处理num个图片
        int batchNumCount = 0;

        INDArray[] featuresMask = null;//特征的掩码数组，通常用于可变长度时间序列模型，可能为空；对于此例来说就是空。
        INDArray[] labelMask = null;//标签的掩码数组，可能为空，通常用于可变长度时间序列。对于此例来说就是空。

        List<MultiDataSet> multiDataSets = new ArrayList<>();//数据集列表

        while (batchNumCount != num && fileIterator.hasNext()) {
            File image = fileIterator.next();//从文件迭代器中取出下一个图片文件
            String imageName = image.getName().substring(0,image.getName().lastIndexOf('.'));//获取图片文件的名字，就是这个图片验证码的真实值        
            
            String[] imageNames = imageName.split("");//将验证码的真实值拆分成单个数字放进数组
            INDArray feature = asMatrix(image);//将图片文件转换成程序能够识别的矩阵
            INDArray[] features = new INDArray[]{feature};//将图片的INDArray转换成INDArray数组
            INDArray[] labels = new INDArray[6];

            Nd4j.getAffinityManager().ensureLocation(feature, AffinityManager.Location.DEVICE);//确保将feature加载到内存
            if (imageName.length() < 6) {	//如果验证码数字个数小于6，则在后面补一个0
                imageName = imageName + "0";
                imageNames = imageName.split("");
            }
            for (int i = 0; i < imageNames.length; i ++) {
                int digit = Integer.parseInt(imageNames[i]);
                labels[i] = Nd4j.zeros(1, 10).putScalar(new int[]{0, digit}, 1);//标签数组的第i个元素为一个1行10列初始化为0的行向量，行向量中的元素如果为1则索引表示标签的这个元素值
            }
            feature =  feature.muli(1.0/255.0);//将图片文件的矩阵元素乘以1/255，得到0-1之间的值

            multiDataSets.add(new MultiDataSet(features, labels, featuresMask, labelMask));

            batchNumCount ++;
        }
        MultiDataSet result = MultiDataSet.merge(multiDataSets);//将一次处理的num张图片的所有数据集合并到一个数据集
        return result;
    }

    public MultiDataSet next(int batchSize) {//批处理batchSize条数据，返回一个MultiDataSet
        try {
            MultiDataSet result = convertDataSet( batchSize );
            return result;
        } catch (Exception e) {
            log.error("the next function shows error", e);
        }
        return null;
    }

    public void reset() {
        load();
    }
    public int totalExamples() {//返回总共有多少张图片
        return numExample;
    }
}
