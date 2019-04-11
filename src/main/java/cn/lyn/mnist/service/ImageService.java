package cn.lyn.mnist.service;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.stereotype.Service;

import cn.lyn.mnist.constant.Constant;
import cn.lyn.mnist.util.CNN_Mnist;

/**
 * 手写数字识别的业务逻辑
 * 
 * @author 亚 南
 *
 */
@Service
public class ImageService {

	/*
	 * 调用工具类训练模型
	 */
	public String tranningOfModel() throws IOException {
		return CNN_Mnist.tranModel();
	}

	/*
	 * 获取训练进度数据
	 */
	public int getProgressNum() {
		return CNN_Mnist.num_progress;
	}

	/*
	 * 图像识别：将图像字节数组读入字节数组输入流，然后读入到图像缓冲区，在调用下面的方法
	 */
	public String imageRecognition(byte[] bytes) throws IOException {
		ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
		BufferedImage bi = ImageIO.read(bais);

		return pictureHandler(bi);
	}

	/*
	 * 图像处理器：将前端读取到的图片缩放成28*28的图片
	 */
	private String pictureHandler(BufferedImage bi) throws IOException {
		// resize to 28*28
		int smallWidth = Constant.smallWidth;
		int smallHeight = Constant.smallHeight;
		Image smallImage = bi.getScaledInstance(smallWidth, smallHeight, Image.SCALE_SMOOTH);
		// 这里一定要使用BufferedImage.TYPE_BYTE_GRAY图像类型 ，保证图像是单通道的灰度图，否则会报错
		BufferedImage bSmallImage = new BufferedImage(smallWidth, smallHeight, BufferedImage.TYPE_BYTE_GRAY);
		Graphics graphics1 = bSmallImage.getGraphics();
		graphics1.drawImage(smallImage, 0, 0, null);
		graphics1.dispose();

		// Use NativeImageLoader to convert to numerical matrix
		NativeImageLoader loader = new NativeImageLoader(28, 28);

		// Get the image into an INDarray
		INDArray image = loader.asMatrix(bSmallImage);

		// 归一化处理，0-255：0-1
		DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
		scaler.transform(image);

		// Pass through to neural Net 加载模型，识别图片，得到一个包含10个可能结果的1行10列数组
		MultiLayerNetwork modelOfNet = ModelSerializer.restoreMultiLayerNetwork("trained_mnist_model.zip");
		INDArray output = modelOfNet.output(image);

		// log.info("## List of Labels in Order## ");
		// 查找10个可能结果的最大值，并记录其索引，索引就是识别结果，max就是准确率大小
		int index = 0;
		double max = 0.000001;
		for (int i = 0; i < output.columns(); i++) {
			if (output.getDouble(0, i) >= max) {
				index = i;
				max = output.getDouble(0, i);
			}
		}

		// 为了测试所用，可以删除下面这段代码
		System.out.println("*****************************" + index);
		System.out.println("-----------------------------" + max);
		File file = new File("./original.png");
		File file2 = new File("./shrink.png");
		ImageIO.write(bi, "png", file);
		ImageIO.write(bSmallImage, "png", file2);

		return "识别结果：" + index + "  准确率：" + max;
	}

}
