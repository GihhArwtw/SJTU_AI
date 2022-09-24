onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /transpose_tb_2/CLK
add wave -noupdate /transpose_tb_2/RST
add wave -noupdate -radix decimal /transpose_tb_2/num
add wave -noupdate /transpose_tb_2/V_in
add wave -noupdate /transpose_tb_2/V_out
add wave -noupdate -radix decimal /transpose_tb_2/data
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {190000 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {176500 ps} {285700 ps}
