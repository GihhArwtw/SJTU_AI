`timescale 1ns / 1ps
module Bayer2RGB_tb;

parameter total_pixel = 296*246;

reg          clk;
reg          rst_n;
wire         cen;
wire         wen;
wire [19:0]  A;
wire [7:0]   D;
wire [7:0]   Q;
wire         O_RGB_data_valid;
wire [7:0]   O_RGB_data_R;          
wire [7:0]   O_RGB_data_G;          
wire [7:0]   O_RGB_data_B;     
wire         start; 

integer i,count;

always #5 clk <= ~clk;

initial begin
   clk = 0;
   rst_n = 0;
   #10;
   rst_n = 1;   
end

// initialize the module Bayer_RGB
Bayer2RGB  Bayer2RGB_inst
(
   .clk(clk),
   .rst_n(rst_n),
   .cen(cen),
   .wen(wen),
   .addr(A),
   .data_out(D),
   .data_in(Q),
   .start(start),
   .O_RGB_data_valid(O_RGB_data_valid),
   .O_RGB_data_R(O_RGB_data_R),
   .O_RGB_data_G(O_RGB_data_G),
   .O_RGB_data_B(O_RGB_data_B)
);


SRAM #(.D_WIDTH(8),.A_WIDTH(20)) sram_inst
(
   .clk(clk),
   .cen(cen),
   .wen(wen),
   .A(A),
   .D(D),
   .Q(Q)
);

//=====================input==================================

integer fin,number_file_R,number_file_G,number_file_B;


initial begin
   number_file_R = $fopen("D:/Textbooks/2021-2022-2/Digital Logic Design/Project/1_pics/31/Bayer2RGB_R.txt","w");
   number_file_G = $fopen("D:/Textbooks/2021-2022-2/Digital Logic Design/Project/1_pics/31/Bayer2RGB_G.txt","w");
   number_file_B = $fopen("D:/Textbooks/2021-2022-2/Digital Logic Design/Project/1_pics/31/Bayer2RGB_B.txt","w");
   i = 0;
   count = 0;
   while (i < total_pixel) begin
      @(posedge clk)
        begin
        if(O_RGB_data_valid)
          begin
            $fwrite(number_file_R,"%2h\n",O_RGB_data_R); 
            $fwrite(number_file_G,"%2h\n",O_RGB_data_G);
            $fwrite(number_file_B,"%2h\n",O_RGB_data_B);
            i = i+ 1;
          end
        end 
      if(start) begin
         count = count + 1;
      end
   end
     #10
          $display("the image Bayer TO RGB is done!!!\n");
          $display("the cost time is : %d",count);
          $fclose(number_file_R);
          $fclose(number_file_G);
          $fclose(number_file_B);
end

endmodule
