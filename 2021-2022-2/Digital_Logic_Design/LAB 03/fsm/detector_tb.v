`timescale 1ns/100ps

module detector_tb;

reg CLK;
reg RST;
reg IN;
wire OUT;

reg[5:0] SEQ;     // in inverse order
reg[5:0] NEXT;    // in inverse order
reg[11:0] CUR;
reg[3:0] count_rst;
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
    count_rst = 0;
    SEQ = 6'b100101;   // in inverse order
    NEXT = 6'b100101;  // in inverse order
    CUR = {NEXT,SEQ};
    count = 12;
join

always #11
begin
    if (count_rst>11)
    begin
        RST = 1;
    end
    else
    begin
        if (count_rst<11)
            RST = 1;
        else
            RST = 0;
        count_rst = count_rst+1; 
    end
end

always #5
begin
    CLK = ~CLK;
end

always #10
begin
    IN = CUR % 2;
    CUR = CUR >> 1;
    count = count-1;
    if (count==0)
    begin
        NEXT = NEXT + 1;
        if (NEXT==0)
            SEQ = SEQ + 1;
        CUR = {NEXT,SEQ};
        count = 12;
    end
end


endmodule