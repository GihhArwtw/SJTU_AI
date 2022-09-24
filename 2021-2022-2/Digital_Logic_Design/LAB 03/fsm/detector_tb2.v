`timescale 1ns/100ps

module detector_tb2;

reg CLK;
reg RST;
reg IN;
wire OUT;

reg[3:0] count;

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
    count = 0;
join

always #5
begin
    CLK = ~CLK;
    RST = 1;
end

always #10
begin
    case (count)
        0: IN = 0;
        1: IN = 1;
        2: IN = 0;
        3: IN = 0;
        default: IN = 1;   //4
    endcase
    count = count + 1;
    if (count>4)
        count = 0;
end

endmodule