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

/**
 * 验证码识别的service层，处理验证码识别相关的业务逻辑
 * 
 * @author 亚 南
 *
 */
@Service
public class VerifyCodeService {

	/*
	 * 训练模型
	 */
	public void ModelOfVerifyCodeReCog() {
		try {
			MultiDigitNumberRecognition.tranningModel();
		} catch (Exception e) {

			e.printStackTrace();
		}
	}

	/*
	 * 更新进度条的数据
	 */
	public int getProgressNum() {
		return MultiDigitNumberRecognition.progressNum;
	}

	// 载入训练集，用于获取图片标签
	private MulRecordDataLoader mulrecordDataLoader = new MulRecordDataLoader("train");

	// 获取图片标签
	public List<String> getLables() {
		return mulrecordDataLoader.getLabels();
	}

	/*
	 * 识别验证码
	 */
	public String recognize(String lable) throws IOException {
		// 根据标签加载验证码图片，并转换成MultiDataSet数据集对象
		MultiDataSet md = ConvertImage(lable);
		// 加载训练好的模型对象
		ComputationGraph model = ModelSerializer.restoreComputationGraph(MultiDigitNumberRecognition.modelPath);
		// 返回识别结果
		return predict(model, md);
	}

	/*
	 * 根据标签，获取验证码图片的位置，然后加载这张图片
	 */
	private MultiDataSet ConvertImage(String imageName) throws IOException {
		// 获取验证码图片文件
		File image = new File(
				this.getClass().getResource("/static/captchaImage/train/" + imageName + ".jpeg").getFile());
		String[] imageNames = imageName.split("");// 将验证码的真实值拆分成数组

		NativeImageLoader loader = new NativeImageLoader(60, 160, 1);// 载入图片高60，宽160，单通道
		INDArray feature = loader.asMatrix(image);// 将图片文件转换成程序能够识别的矩阵
		INDArray[] features = new INDArray[] { feature };// 将图片的INDArray转换成INDArray数组
		INDArray[] labels = new INDArray[6];// 图片标签数组，共6个数组，每个数组都是INDArray

		Nd4j.getAffinityManager().ensureLocation(feature, AffinityManager.Location.DEVICE);// 确保将feature加载到内存
		if (imageName.length() < 6) { // 如果验证码数字个数小于6，则在后面补一个0
			imageName = imageName + "0";
			imageNames = imageName.split("");
		}
		for (int i = 0; i < imageNames.length; i++) {
			int digit = Integer.parseInt(imageNames[i]);
			// 标签数组的第i个元素为一个1行10列初始化为0的行向量，行向量中的元素如果为1则索引表示标签的这个元素值
			labels[i] = Nd4j.zeros(1, 10).putScalar(new int[] { 0, digit }, 1);
		}
		feature = feature.muli(1.0 / 255.0);// 将图片文件的矩阵元素乘以1/255，得到0-1之间的值

		MultiDataSet result = new MultiDataSet(features, labels, null, null);
		return result;
	}

	private String predict(ComputationGraph model, MultiDataSet md) {
		INDArray[] output = model.output(md.getFeatures());// 调用MultiDataSet的getFeatures()方法，获取图片的输入特征数组，然后用模型获得每个特征的识别结果的可能性数组
		String peLabel = "";// 用于拼凑识别结果的字符串
		INDArray preOutput = null;// 存储每个位置上10个可能结果的大小的数组，1行10列的一个数组

		for (int digit = 0; digit < 6; digit++) {// 总共有6个数字
			preOutput = output[digit];// 获取第digit位的10个可能结果的可能性大小数组
			peLabel += Nd4j.argMax(preOutput, 1).getInt(0);// 在peLable上拼上预测的结果的可能性最大的值
		}
		System.out.println(peLabel);
		return peLabel;
	}
}
