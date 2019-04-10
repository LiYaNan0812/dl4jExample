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

@Service
public class ImageService {

    public String tranningOfModel() throws IOException {

        return CNN_Mnist.tranModel();
    }
    
    public int getProgressNum() {
    	return CNN_Mnist.num_progress;
    }

    public String imageRecognition(byte[] bytes) throws IOException {
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        BufferedImage bi = ImageIO.read(bais);
        
       
        return pictureHandler(bi);
    }


    private String pictureHandler(BufferedImage bi) throws IOException {
        //resize to 28*28
        int smallWidth = Constant.smallWidth;
        int smallHeight = Constant.smallHeight;
        Image smallImage = bi.getScaledInstance(smallWidth, smallHeight, Image.SCALE_SMOOTH);
      //这里一定要使用BufferedImage.TYPE_BYTE_GRAY图像类型 ，保证图像是单通道的灰度图，否则会报错
        BufferedImage bSmallImage = new BufferedImage(smallWidth, smallHeight,BufferedImage.TYPE_BYTE_GRAY);
        Graphics graphics1 = bSmallImage.getGraphics();
        graphics1.drawImage(smallImage, 0, 0, null);
        graphics1.dispose();
        
       
        

        // Use NativeImageLoader to convert to numerical matrix
        NativeImageLoader loader = new NativeImageLoader(28, 28);

        // Get the image into an INDarray
        INDArray image = loader.asMatrix(bSmallImage);
        // 0-255
        // 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        
        
        // Pass through to neural Net
        MultiLayerNetwork modelOfNet = ModelSerializer.restoreMultiLayerNetwork("trained_mnist_model.zip");
      // System.out.println(modelOfNet.predict(image));
        INDArray output = modelOfNet.output(image);
        

        //log.info("## List of Labels in Order## ");
        // In new versions labels are always in order
        int index = 0;
        double max = 0.000001;
        for (int i = 0; i < output.columns(); i++) {
            if (output.getDouble(0, i) >= max) {
                index = i;
                max = output.getDouble(0, i);
            }
        }

        System.out.println("*****************************" + index);
        System.out.println("-----------------------------" + max);
        File file = new File("./original.png");
        File file2 = new File("./shrink.png");
        ImageIO.write(bi, "png", file);
        ImageIO.write(bSmallImage, "png", file2);

        return "识别结果：" + index + "  准确率：" + max;
    }

	


}
