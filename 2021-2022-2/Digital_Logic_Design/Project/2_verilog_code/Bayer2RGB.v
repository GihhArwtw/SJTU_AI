`timescale 1ns / 1ps
module Bayer2RGB 
(
                 input              clk,
                 input              rst_n,
                 input     [7:0]    data_in,
                 output reg         cen,
                 output reg         wen,
                 output reg[19:0]   addr,
                 output    [7:0]    data_out,
                 output reg         start,
                 output reg         O_RGB_data_valid,
                 output reg[7:0]    O_RGB_data_R,
                 output reg[7:0]    O_RGB_data_G,
                 output reg[7:0]    O_RGB_data_B
    );

parameter column    = 245;     // maximum column index, i.e. #(columns)-1
parameter precolumn = 244;     // #(column)-2;
parameter rows      = 295;     // maximum row index, i.e. #(rows)-1;
parameter total     = 296*246; // #(pixels)

reg       mask;             // odd column / even column
reg       pattern;          // mask=0:  0-B, 1-G;  mask=1:  0-G, 1-R
reg[8:0]  line;             // how many lines we have processed
reg[7:0]  count;            // output the count-th column
reg[1:0]  fetch_count;      // fetch four numbers for one unit
reg[7:0]  pixel_r;
reg[8:0]  pixel_g;          // note there might be carry in the addition!
reg[7:0]  pixel_b;
reg       hold;
reg[7:0]  r_[(column>>1):0];
reg[7:0]  g_[(column>>1):0];
reg[7:0]  b_[(column>>1):0];

always @(posedge clk) begin
   if(~cen) begin
      start <= 1;
   end
end

always @(negedge rst_n) 
fork
   pattern <= 0;
   mask <= 0;
   cen <= 1;
   wen <= 1;
   O_RGB_data_valid <= 0;
   fetch_count <= 2'b00;
join

always @(posedge rst_n)
fork
   addr <= 0;
   count <= 1;
   line <= 0;
   hold <= 1;
   cen <= 0;
join

// _________________________________________________________
// Note that [addr] =(1clk)=> [data_in] =(1clk)=> pixel_

always @(posedge clk)
begin
   if (hold)          // hold state
   begin
     addr <= 1;
	  hold <= 0;
   end
   else
   if (line>rows)     // end state
   begin
      cen <= 1;
      O_RGB_data_valid <= 0;
   end
   else if (rst_n)
   begin
      case ({mask,pattern,fetch_count})   // to use fewer adders
         4'b0000:          addr <= addr+column;
         4'b0010:          addr <= addr-column;
         4'b0001, 4'b0011: addr <= addr+1;
      endcase
      case ({mask,pattern})
         2'b00:begin
                  case(fetch_count)
                     2'b00:begin
                              pixel_b <= data_in;
                              O_RGB_data_valid <= 0;
                           end
                     2'b01:begin
                              pixel_g <= data_in;
                              O_RGB_data_valid <= 0;
                           end
                     2'b10:begin
                              pixel_g <= pixel_g+data_in;
                              O_RGB_data_valid <= 0;
                           end
                     2'b11:begin
                              pixel_r <= data_in;
                              cen <= 1;
                              O_RGB_data_valid <= 1;
                              O_RGB_data_R <= data_in;
                              O_RGB_data_G <= pixel_g[8:1];
                              O_RGB_data_B <= pixel_b;
                              r_[count>>1] <= data_in;
                              g_[count>>1] <= pixel_g[8:1];
                              b_[count>>1] <= pixel_b;
                              pattern <= pattern+1;
                              count <= count+1;
                           end
                  endcase
                  fetch_count <= fetch_count+1;
               end
         2'b01:begin
                  O_RGB_data_valid <= 1;
                  pattern <= pattern+1;
                  count <= count+1;
                  cen <= 0;
                  if (count>precolumn)
                  begin
                     mask <= mask+1;
                     count <= 0;
                     line <= line+1;
                     cen <= 1;
                  end
               end
         default:
               begin
                  O_RGB_data_valid <= 1;
                  O_RGB_data_R <= r_[count>>1];
                  O_RGB_data_G <= g_[count>>1];
                  O_RGB_data_B <= b_[count>>1];
                  count <= count+1;
                  if (count>=precolumn)
                  begin
                     if (count<column)
                     begin
                        addr <= addr+column;
                        if (addr+column>=total)   cen <= 1;
                        else  cen <= 0;
                     end
                     else
                     begin
                        addr <= addr+1;
                        count <= 1;
                        mask <= mask+1;
                        line <= line+1;
                        fetch_count <= 2'b00; 
                     end
                     if (addr>=total)  cen <= 1;
                  end
               end
      endcase
   end
end

endmodule


// Need 127429 CLKS to complete the whole process.