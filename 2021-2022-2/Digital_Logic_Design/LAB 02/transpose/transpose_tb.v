`timescale 1ns/100ps

module transpose_tb;

reg CLK;
reg RST;
reg[31:0] num;
reg[4:0] count_rst;
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
    count_rst = 0;
    V_in = 0;
join

always #5
begin
    CLK = ~CLK;
end

always #10
begin
    num = $random % 2147483648;
    if ({$random}%10>2)
        V_in = 1;
    else   
        V_in = 0;
end

always #21                    // to test RST (for only twice)
begin
    case (count_rst)
        12:      RST = 1;
        11:      begin
                    RST = 0;
                    count_rst = count_rst+1;
                 end
        default: begin
                    RST = 1;
                    count_rst = count_rst+1;
                 end
    endcase
end

endmodule