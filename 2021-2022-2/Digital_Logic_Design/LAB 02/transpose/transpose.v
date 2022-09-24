`timescale 1ns/100ps

module transpose(
    input wire clk,
    input wire rst_n,
    input wire[31:0] data_i,
    input wire valid_i,
    output reg[31:0] data_o,
    output reg valid_o
);

integer inp[8:0];
integer tmp[8:0];
integer out[8:0];

reg[3:0] count_in;
reg[3:0] count_out;

always @(negedge rst_n)
begin
    count_in = 0;
    count_out = 0;
    valid_o = 0;
    data_o = 0;
end

always @(posedge clk)
begin
    if (rst_n==1)
    begin
        if (count_out==0)
        begin
            valid_o = 0;
        end
        else
        begin
            data_o = out[count_out-1];
            valid_o = 1;
            count_out = count_out+1;
            if (count_out>9)
                count_out = 0;
        end
        if (valid_i==1)
        begin
            inp[count_in%3*3+count_in/3] = data_i;
            count_in = count_in+1;
        end
        if (count_in==9)
        begin
            out[0] <= inp[0];
            out[1] <= inp[1];
            out[2] <= inp[2];
            out[3] <= inp[3];
            out[4] <= inp[4];
            out[5] <= inp[5];
            out[6] <= inp[6];
            out[7] <= inp[7]; 
            out[8] <= inp[8];
            count_out = 1;
            count_in = 0;
        end
    end
end

endmodule