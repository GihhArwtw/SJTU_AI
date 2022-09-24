`timescale 1ns/100ps

module detector_tb_rand;

reg CLK;
reg RST;
reg IN;
wire OUT;

detector DTR(
    .clk(CLK),
    .rst_n(RST),
    .din(IN),
    .detector(OUT)
);

initial fork
    CLK = 1;
    RST = 0;
    IN = 1;
join

always #5
begin
    CLK = ~CLK;
    RST = 1;
end

always #10
begin
    IN = {$random}%2;
end

endmodule