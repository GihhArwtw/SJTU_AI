onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /transpose_tb/CLK
add wave -noupdate /transpose_tb/RST
add wave -noupdate -radix decimal /transpose_tb/num
add wave -noupdate /transpose_tb/V_in
add wave -noupdate /transpose_tb/V_out
add wave -noupdate -radix decimal /transpose_tb/data
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {480000 ps} 0}
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
WaveRestoreZoom {477400 ps} {591100 ps}
