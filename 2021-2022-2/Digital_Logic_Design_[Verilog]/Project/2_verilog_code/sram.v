`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2022/05/16 13:39:52
// Design Name: 
// Module Name: sram
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module SRAM #(parameter D_WIDTH = 8,A_WIDTH = 4)
(
input clk,
input cen,
input wen,
input [A_WIDTH-1:0] A,
input [D_WIDTH-1:0] D,
output reg [D_WIDTH-1:0] Q
);
 
reg [D_WIDTH-1 :0] mem [2**A_WIDTH-1:0];

initial begin
    $readmemh("D:/Textbooks/2021-2022-2/Digital Logic Design/Project/1_pics/31/img.txt",mem);
end

always @(posedge clk)begin
if(!cen & !wen)
    mem[A]  <=  D;
end
 
always @(posedge clk)begin
if(wen  & (!cen))
    Q   <=  mem[A];
end
endmodule

