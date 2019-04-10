package cn.lyn.mnist.service;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;


import cn.lyn.verificationCode.util.MulRecordDataLoader;
import cn.lyn.verificationCode.util.MultiDigitNumberRecognition;

@Service
public class VerifyCodeService {
	
	private MulRecordDataLoader mulrecordDataLoader = new MulRecordDataLoader("train");
	//训练模型
	public void ModelOfVerifyCodeReCog() {		
		try {
			MultiDigitNumberRecognition.tranningModel();
		} catch (Exception e) {
			
			e.printStackTrace();
		}
	}


	//更新进度条的数据
	public int getProgressNum() {
	
		return MultiDigitNumberRecognition.progressNum;
	}
	
	//获取图片标签
	public List<String> getLables(){
		
		return mulrecordDataLoader.getLabels();
	}

	public String recognize(String lable) throws IOException {
		
			MultiDataSet md = ConvertImage(lable);
			ComputationGraph model = ModelSerializer.restoreComputationGraph(MultiDigitNumberRecognition.modelPath);
			return predict(model,md);
		
	}
	
	private MultiDataSet ConvertImage(String imageName) throws IOException {
		File image = new File(this.getClass().getResource("/static/captchaImage/train/"+imageName+".jpeg").getFile());//获取验证码图片文件
		String[] imageNames = imageName.split("");//将验证码的真实值拆分成数组
		
		NativeImageLoader loader = new NativeImageLoader(60,160,1);
		 INDArray feature = loader.asMatrix(image);//将图片文件转换成程序能够识别的矩阵
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
         List<MultiDataSet> multiDataSets = new ArrayList<>();//数据集列表
         MultiDataSet e = new MultiDataSet(features, labels,null,null);
         multiDataSets.add(e);
         MultiDataSet result = MultiDataSet.merge(multiDataSets);//将一次处理的num张图片的所有数据集合并到一个数据集
         return result;
         
	}
	
	private String predict(ComputationGraph model,MultiDataSet md) {
		 INDArray[]  output = model.output(md.getFeatures());
		  String peLabel = "";
		  INDArray preOutput = null;
        
		  for (int digit = 0; digit < 6; digit ++) {//总共有6个数字
              preOutput = output[digit];
              peLabel += Nd4j.argMax(preOutput, 1).getInt(0);//在peLable上拼上预测的结果的可能性最大的值
             
          }
		  System.out.println(peLabel);
		return peLabel;
		
	}
}
