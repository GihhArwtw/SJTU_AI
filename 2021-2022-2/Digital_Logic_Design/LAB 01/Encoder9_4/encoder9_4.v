`timescale 1ns/100ps

module encoder9_4(a,y);

input [8:0] a;
output[3:0] y;

wire[8:0] a;
reg[3:0] y;

always @(a)
begin
    if (a[8]==1)
        y = 4'b1001;
    else if (a[7]==1)
        y = 4'b1000;
    else if (a[6]==1)
        y = 4'b0111;
    else if (a[5]==1)
        y = 4'b0110;
    else if (a[4]==1)
        y = 4'b0101;
    else if (a[3]==1)
        y = 4'b0100;
    else if (a[2]==1)
        y = 4'b0011;
    else if (a[1]==1)
        y = 4'b0010;
    else if (a[0]==1)
        y = 4'b0001;
    else
        y = 4'b0000;
end


endmodule