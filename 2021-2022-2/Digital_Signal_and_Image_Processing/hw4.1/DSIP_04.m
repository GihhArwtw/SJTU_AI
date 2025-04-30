audio_file_path = "./宇多田ヒカル - 夕凪.mp3";
audio_name = "夕凪";


[y,Fs] = audioread(audio_file_path);
duration = length(y);
chn = ["Channel: Surrounding Left","Channel: Surrounding Right"];


% TASK 01

title_ = ["The Waveform of "+audio_name];
file_name = ["waveform_"+audio_name+"_orig.bmp"];
Waveform(y,2,chn,1800,450,title_,file_name);

title_ = ["The Spectrogram of "+audio_name];
file_name = ["specgram_"+audio_name+"_orig.bmp"];
STFT_display(y,Fs,2,chn,1800,500,title_,file_name);


% TASK 02
y_5 = our_downsampling(y, Fs, 5000);
y_10 = our_downsampling(y, Fs, 10000);
y_15 = our_downsampling(y, Fs, 15000);

file_name = [audio_name+"_5kHz_downspl.wav"];
audiowrite(file_name, y_5, 5000);
file_name = [audio_name+"_10kHz_downspl.wav"];
audiowrite(file_name, y_10, 10000);
file_name = [audio_name+"_15kHz_downspl.wav"];
audiowrite(file_name, y_15, 15000);


chn = ["Channel 'Surrounding Left' Downsampled by 5kHz",...
       "Channel 'Surrounding Left' Downsampled by 10kHz",...
       "Channel 'Surrounding Left' Downsampled by 15kHz"];

title_ = ["The Waveform of "+audio_name+" Downsampled by Different Frequencies"];
file_name = ["waveform_"+audio_name+"_downspl.bmp"];
Waveform_3(y_5(:,1),y_10(:,1),y_15(:,1),chn,1800,725,title_,file_name);

title_ = ["The Spectrogram of "+audio_name+" Downsampled by Different Frequencies"];
file_name = ["specgram_"+audio_name+"_downspl.bmp"];
STFT_display_3(y_5(:,1),y_10(:,1),y_15(:,1),Fs,chn,1800,725,title_,file_name);

psg_begin = 535000;
psg_end = 535999;
psg_1 = y_5(psg_begin:psg_end,1);
psg_2 = y_10(psg_begin*2:psg_end*2,1);
psg_3 = y_15(psg_begin*3:psg_end*3,1);
title_ = ["The Waveform of the Same Part of "+audio_name+" Downsampled by Different Frequencies"];
file_name = ["waveform_"+audio_name+"_downspl_partial.bmp"];
fig = Waveform_3(psg_1,psg_2,psg_3,chn,1800,725,title_,file_name);


% TASK 03
x = transpose(linspace(0,duration,duration));

y_5_st = Interpolate(x, y_5, 2, Fs, 5000, duration, "linear");
y_10_st = Interpolate(x, y_10, 2, Fs, 10000, duration, "linear");
y_15_st = Interpolate(x, y_15, 2, Fs, 15000, duration, "linear");

chn = ["Restoration of Channel 'Surrounding Left' Downsampled by 5kHz",...
       "Restoration of Channel 'Surrounding Left' Downsampled by 10kHz",...
       "Restoration of Channel 'Surrounding Left' Downsampled by 15kHz"];
title_ = ["The Waveform of the Linear Interpolation of "+audio_name+" Downsampled."];
file_name = ["waveform_"+audio_name+"_rstd_linear.bmp"];
Waveform_3(y_5_st(:,1),y_10_st(:,1),y_15_st(:,1),chn,1800,725,title_,file_name);

title_ = ["The Spectrogram of the Linear Interpolation of "+audio_name+" Downsampled."];
file_name = ["specgram_"+audio_name+"_rstd_linear.bmp"];
STFT_display_3(y_5_st(:,1),y_10_st(:,1),y_15_st(:,1),Fs,chn,1800,725,title_,file_name);

file_name = [audio_name+"_5kHz_linear.wav"];
audiowrite(file_name, y_5_st, Fs);
file_name = [audio_name+"_10kHz_linear.wav"];
audiowrite(file_name, y_10_st, Fs);
file_name = [audio_name+"_15kHz_linear.wav"];
audiowrite(file_name, y_15_st, Fs);


y_5_st = Interpolate(x, y_5, 2, Fs, 5000, duration, "spline");
y_10_st = Interpolate(x, y_10, 2, Fs, 10000, duration, "spline");
y_15_st = Interpolate(x, y_15, 2, Fs, 15000, duration, "spline");

chn = ["Restoration of Channel 'Surrounding Left' Downsampled by 5kHz",...
       "Restoration of Channel 'Surrounding Left' Downsampled by 10kHz",...
       "Restoration of Channel 'Surrounding Left' Downsampled by 15kHz"];
title_ = ["The Waveform of the Spline Interpolation of "+audio_name+" Downsampled."];
file_name = ["waveform_"+audio_name+"_rstd_spline.bmp"];
Waveform_3(y_5_st(:,1),y_10_st(:,1),y_15_st(:,1),chn,1800,725,title_,file_name);

title_ = ["The Spectrogram of the Spline Interpolation of "+audio_name+" Downsampled."];
file_name = ["specgram_"+audio_name+"_rstd_spline.bmp"];
STFT_display_3(y_5_st(:,1),y_10_st(:,1),y_15_st(:,1),Fs,chn,1800,725,title_,file_name);

file_name = [audio_name+"_5kHz_spline.wav"];
audiowrite(file_name, y_5_st, Fs);
file_name = [audio_name+"_10kHz_spline.wav"];
audiowrite(file_name, y_10_st, Fs);
file_name = [audio_name+"_15kHz_spline.wav"];
audiowrite(file_name, y_15_st, Fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = Waveform( y, n, labels, scale_x, scale_y, title_, file_name )
    fig = figure("Position",[0.1,0.1,scale_x,scale_y]);
    for i = 1:n
        subplot(n,1,i), plot(y(:,i),'LineWidth',1.5);
        xlabel('Time','FontName','serif','FontSize',12);
        ylabel('Magnitude','FontName','serif','FontSize',12);
        title(labels(i),'FontName','serif','FontSize',12);
    end
    sgtitle(title_,'FontName','serif','FontSize',15);
    saveas(fig,file_name);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = STFT_display( y, Fs, n, labels, scale_x, scale_y, title_, file_name )
    fig = figure("Position",[0.1,0.1,scale_x,scale_y]);
    for i = 1:n
        subplot(n,1,i), stft(y(:,i),Fs);
        xlabel('Time','FontName','serif','FontSize',12);
        ylabel('Frequency (kHz)','FontName','serif','FontSize',12);
        title(labels(i),'FontName','serif','FontSize',12);
    end
    sgtitle(title_,'FontName','serif','FontSize',15);
    saveas(fig,file_name);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y] = our_downsampling( x, orig_sample_rate, targ_sample_rate )
    if (mod(orig_sample_rate,targ_sample_rate)==0)
        y = downsample(x, orig_sample_rate/targ_sample_rate);
    else
        y = resample(x, targ_sample_rate, orig_sample_rate);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fig] = Waveform_3( y1, y2, y3, labels, scale_x, scale_y, title_, file_name )
    fig = figure("Position",[0.1,0.1,scale_x,scale_y]);
    subplot(3,1,1), plot(y1,'LineWidth',1.5);
    xlabel('Time','FontName','serif','FontSize',12);
    ylabel('Magnitude','FontName','serif','FontSize',12);
    title(labels(1),'FontName','serif','FontSize',12);
    subplot(3,1,2), plot(y2,'LineWidth',1.5);
    xlabel('Time','FontName','serif','FontSize',12);
    ylabel('Magnitude','FontName','serif','FontSize',12);
    title(labels(2),'FontName','serif','FontSize',12);
    subplot(3,1,3), plot(y3,'LineWidth',1.5);
    xlabel('Time','FontName','serif','FontSize',12);
    ylabel('Magnitude','FontName','serif','FontSize',12);
    title(labels(3),'FontName','serif','FontSize',12);
    sgtitle(title_,'FontName','serif','FontSize',15);
    saveas(fig,file_name);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fig] = STFT_display_3( y1, y2, y3, Fs, labels, scale_x, scale_y, title_, file_name )
    fig = figure("Position",[0.1,0.1,scale_x,scale_y]);
    subplot(3,1,1), stft(y1,Fs);
    xlabel('Time','FontName','serif','FontSize',12);
    ylabel('Frequency (kHz)','FontName','serif','FontSize',12);
    title(labels(1),'FontName','serif','FontSize',12);
    subplot(3,1,2), stft(y2,Fs);
    xlabel('Time','FontName','serif','FontSize',12);
    ylabel('Frequency (kHz)','FontName','serif','FontSize',12);
    title(labels(2),'FontName','serif','FontSize',12);
    subplot(3,1,3), stft(y3,Fs);
    xlabel('Time','FontName','serif','FontSize',12);
    ylabel('Frequency (kHz)','FontName','serif','FontSize',12);
    title(labels(3),'FontName','serif','FontSize',12);
    sgtitle(title_,'FontName','serif','FontSize',15);
    saveas(fig,file_name);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y] = Interpolate( samples, x, n, Fs, f, duration, Method)
    z = griddedInterpolant(0:(double(Fs)/double(f)):duration,x(:,1),Method);
    y = z(samples);
    for i = 2 : n
        z = griddedInterpolant(0:(double(Fs)/double(f)):duration,x(:,i),Method);
        y = cat(2,y,z(samples));
    end
end