package cn.lyn.mnist.controller;

import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import cn.lyn.mnist.service.ImageService;
import sun.misc.BASE64Decoder;
/**
 * 处理有关手写数字识别相关的请求的处理器
 * @author 亚 南
 *
 */
@Controller
@RequestMapping("/")
public class RequestController {
    @Autowired
    private ImageService imageService;//注入imageService对象

    /*
     * 当请求路径是/tranningOfModel时，开启训练模型
     */
    @RequestMapping("tranningOfModel")
    @ResponseBody
    public String tranningOfModel() {
        try {

            String evalStats = imageService.tranningOfModel();
            int startIndex = evalStats.lastIndexOf("========================Evaluation Metrics========================");
            int endIndex = evalStats.lastIndexOf("Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)");
            String str = ""; 
            str=evalStats.substring(startIndex, endIndex);
          
            return str.substring(str.indexOf("Accuracy"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    /*
     * 访问路径是/imageRecognition，将传递过来的图片数据解码成字节数粗，然后调用service进行图像识别
     */
    @RequestMapping("imageRecognition")
    @ResponseBody
    public String imageRecognition(@RequestParam String picture) throws IOException {

        BASE64Decoder decoder = new BASE64Decoder();//url解码前端页面传过来的数据
        byte[] bytes = decoder.decodeBuffer(picture);
  
        return imageService.imageRecognition(bytes);
    }

    /*
     * 请求路径/mnist,转发页面到mnist.html
     */
    @RequestMapping("mnist")
    public String mnist() {
    	return "mnist";
    }
    
    /*
     * 请求路径是/index，转发页面到index.html
     */
    @RequestMapping("index")
    public String index() {
        return "index";
    }
    
    /*
     * 请求路径是/show,转发页面到show.html
     */
    @RequestMapping("show")
    public String show() {
        return "show";
    }
    
    /*
     * 请求路径是/getProgressNum时，返回一个有关训练进度的数据
     */
    @RequestMapping("getProgressNum")
    @ResponseBody
    public String getProgressNum() {
    	int num = imageService.getProgressNum();
    	return String.valueOf(num);
    }
    
}
