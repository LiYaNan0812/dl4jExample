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

@Controller
@RequestMapping("/verifyCode/")
public class VerifyCodeController {
	@Autowired
	private VerifyCodeService verifyCodeService;
	
	@RequestMapping("verifyCodeRecognition")
    public String verifyCodeRecognition() {
    	return "verifyCodeRecognition";
    }
	
	@RequestMapping("verifyCodeRecognizePage")
	public String verifyCodeRecognizePage(Model model) {
	
		if(!model.containsAttribute("lables")) {
			List<String> lables = verifyCodeService.getLables();
			model.addAttribute("lables", lables);
		}
		
		return "verifyCodeRecognizePage";
	}
    
    @RequestMapping("ModelOfVerifyCodeRecog")
    public String ModelOfVerifyCodeRecog(){
    	verifyCodeService.ModelOfVerifyCodeReCog();
    	return "verifyCodeRecognition";
    }
    
    @RequestMapping("recognize")
    @ResponseBody
    public String recognize(@RequestParam String lable) throws IOException {
    	
    	return verifyCodeService.recognize(lable);
    }
    
    
    
    @RequestMapping("getProgressNum")
    @ResponseBody
    public String getProgressNum() {
    	int num = verifyCodeService.getProgressNum();
    	System.out.println(num);
    	return String.valueOf(num);
    }
}
