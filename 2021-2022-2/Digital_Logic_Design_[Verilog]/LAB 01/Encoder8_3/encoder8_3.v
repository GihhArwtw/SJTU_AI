`timescale 1ns/100ps

module encoder8_3(a,y);

input [7:0] a;
output[2:0] y;

wire[7:0] a;
reg[2:0] y;

// 原本想搞一搞数据流建模的（如下），然而多了非法输入默认值0这个要求之后就只好行为级建模用case语句了。
/*
assign y[0] = a[1] | a[3] | a[5] | a[7];
assign y[1] = a[2] | a[3] | a[6] | a[7];
assign y[2] = a[4] | a[5] | a[6] | a[7];
*/

//所以现在用行为级建模case语句了，如下。
always @(a)
begin
  case (a)
    8'b00000010: y=3'b001;
    8'b00000100: y=3'b010;
    8'b00001000: y=3'b011;
    'b00010000: y=3'b100;
    8'b00100000: y=3'b101;
    8'b01000000: y=3'b110;
    8'b10000000: y=3'b111;
    default:     y=3'b000;    // including the case of a=8'b00000001.
  endcase
end


endmodule