`timescale 1ns/100ps

module countdown_tb;

reg CLK;
reg LD;
reg[2:0] count;
reg[9:0] DATA;
wire OUT;

countdown Counter(
    .clk(CLK),
    .load(LD),
    .data_load(DATA),
    .tc(OUT)
);

initial fork
    CLK = 0;
    LD = 0;
    DATA = 0;
    count = 1;
join

always #5
begin
    CLK = ~CLK;
end

always #6
begin
    LD = {$random}%2;
end

always #50             // if the interval become shorter,
                       // it is highly likely that the counter is loading
                       // "data_load" all the time and never counts down.
begin
    if (count==3)
    begin
        DATA <= 0;
        count <= 0;
    end
    else
    begin
        DATA <= {$random}%8;    //DATA = {$random}%1024 for complete test;
        count <= count+1;
    end
end


endmodule