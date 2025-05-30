# 基于本地知识库构建RAG应用



### 启动

Run: uvicorn main:app --port 7866 --reload
Then visit 127.0.0.1:7866



![image-20250530125646305](images\image-20250530125646305.png)



### 异常处理

![image-20250530174604611](images\image-20250530174604611.png)



修改了gradio的版本

![2ceff04b860f1890428b1df1c850d89](C:\Users\15761\Documents\WeChat Files\wxid_2b7yqely3nd421\FileStorage\Temp\2ceff04b860f1890428b1df1c850d89.png)



不报错的，但是上传文件提示Internal Server Error

![image-20250530185909536](images\image-20250530185909536.png)



![image-20250530185949363](images\image-20250530185949363.png)

![image-20250530190030859](images\image-20250530190030859.png)



参考文档：https://metaso.cn/search/8616742959721373696?q=TypeError%3A+argument+of+type+%27bool%27+is+not+iterable%0D%0AINFO%3A+++++127.0.0.1%3A4696+-+%22GET+%2Fupload_data%2Finf



重启以后好了