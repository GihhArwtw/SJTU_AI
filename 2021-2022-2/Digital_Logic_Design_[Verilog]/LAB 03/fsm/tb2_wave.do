onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /detector_tb2/CLK
add wave -noupdate /detector_tb2/RST
add wave -noupdate /detector_tb2/IN
add wave -noupdate /detector_tb2/OUT
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {1819200 ps} 0}
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
WaveRestoreZoom {1789900 ps} {1989800 ps}
