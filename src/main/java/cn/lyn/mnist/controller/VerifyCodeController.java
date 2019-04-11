package cn.lyn.mnist.controller;

import java.io.IOException;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import cn.lyn.mnist.service.VerifyCodeService;
/**
 * 验证码识别的Controller,处理有关验证码识别的请求
 * @author 亚 南
 *
 */
@Controller
@RequestMapping("/verifyCode/")
public class VerifyCodeController {
	
	@Autowired
	private VerifyCodeService verifyCodeService;//使用spring注解的方式注入业务层对象
	
	/*
	 * 访问/verifyCode/verifyCodeRecognition路径时将页面转发到templates/verifyCodeREcognition.html页面
	 */
	@RequestMapping("verifyCodeRecognition")
    public String verifyCodeRecognition() {
    	return "verifyCodeRecognition";
    }
	
	/*
	 * 请求路径是/verifyCode/verifyCodeREcognizePage时，获取训练集的图片标签集合放入model中，然后将页面转发到
	 * /templates/verifyCodeRecognizePage页面
	 */
	@RequestMapping("verifyCodeRecognizePage")
	public String verifyCodeRecognizePage(Model model) {
	
		if(!model.containsAttribute("lables")) {
			List<String> lables = verifyCodeService.getLables();
			model.addAttribute("lables", lables);
		}
		
		return "verifyCodeRecognizePage";
	}
    
	/*
	 * 请求路径是/verifyCode/ModelOfVerifyCodeRecog时调用service层的方法，开启训练模型 
	 */
    @RequestMapping("ModelOfVerifyCodeRecog")
    public String ModelOfVerifyCodeRecog(){
    	verifyCodeService.ModelOfVerifyCodeReCog();
    	return "verifyCodeRecognition";
    }
    
    /*
     * 请求路径是/verifyCode/recognize时，会以json字符串的形式返回验证码的识别结果
     * 传入的参数lable是用来查找图片的位置的，返回的识别结果是用模型识别出来的
     */
    @RequestMapping("recognize")
    @ResponseBody
    public String recognize(@RequestParam String lable) throws IOException {
    	
    	return verifyCodeService.recognize(lable);
    }
    
    
    /*
     * 请求路径是/verifyCode/getProgressNum时，会获取一个有关当前训练进度的数据
     */
    @RequestMapping("getProgressNum")
    @ResponseBody
    public String getProgressNum() {
    	int num = verifyCodeService.getProgressNum();
    	return String.valueOf(num);
    }
}
