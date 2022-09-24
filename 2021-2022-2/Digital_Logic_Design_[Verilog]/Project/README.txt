***********************************************************
* 2022 数字逻辑课程设计——课程项目
* Bayer 格式到RGB格式的转换
***********************************************************

1. pics. 包含34个不同的图像数据。
    * format_and_size.txt 包含图像尺寸信息以及使用的贝尔图像格式。
    * img.png 图像信息，可用于对比的正确性。
    * img.txt   原始bayer格式图像，同时也是Verilog代码的输入文件。

2. verilog_code. 包含提供的testbench以及电路接口。
    * Bayer2RGB_tb.v  顶层testbench文件，包含数据的读取以及待撰写模块与SRAM之间的交互
    * SRAM.v  原始图像数据会先存储到SRAM中，电路模块会从SRAM中读取数据进行处理
    * Bayer2RGB.v  转换电路，接口已预先定义好，如非必要请勿修改。

3. img_display. 包含显示图像的matlab代码。
    * bayer_img_display.m  显示bayer格式图像
    * rgb_img_display.m  显示RGB格式图像
     运行两个代码时，对应的路径需要修改

4. 实验图像分配.xls 
  * 各位同学分配到的实验图像

5. Project 说明文档.pdf
    *  实验说明文档

