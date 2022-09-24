`timescale 1ns/100ps

module transpose_tb_2;

reg CLK;
reg RST;
reg[31:0] num;
reg V_in;

wire V_out;
wire[31:0] data;

transpose MAT(
    .clk(CLK),
    .rst_n(RST),
    .data_i(num),
    .valid_i(V_in),
    .data_o(data),
    .valid_o(V_out)
);

initial fork
    CLK = 1;
    RST = 0;
    V_in = 0;
join

always #5
begin
    CLK = ~CLK;
    RST = 1;
end

always #10
begin
    num = $random % 2147483648;
    V_in = 1;
end

endmodule