t_s = 0.001; %0.05 %0.2
f_s = 1/t_s;
Len = 10;
n = Len/t_s;
N = n+5/t_s;

x_ = -1;
x = [0];
y = [0];
zero = 1/t_s + 1;
for i=0:(zero+N)
    x_ = x_ + t_s;
    x(i+1) = x_;
    y(i+1) = recwindow(x_);
end

% Task 01
fig = figure,
subplot(2,1,1),plot(x,y,'LineWidth',1.5);     %stem(x,y,'filled');
title('Sequence in Time Domain','FontSize',12);
xlabel('Time');
ylabel('Amplitude');
y_f = fftshift(abs(fft(y(zero:zero+n),N+1)));
subplot(2,1,2),plot(2*(-N/2:N/2)/N,y_f,'LineWidth',1.5);   %stem(x,y_f,'filled');
title('Sequence in Frequency Domain','FontSize',12);
xlabel('x π rad/sample, Frequency');
ylabel('Amplitude');
title_ = ["T_s="+num2str(t_s)];
sgtitle(title_);
file_name = ["SamplingRecWin_Ts="+num2str(t_s)+"_1.bmp"];
saveas(fig,file_name);

% Task 02
x_ = -1;
x = [0];
y = [0];
for i=0:(zero+N)
    x_ = x_ + t_s;
    x(i+1) = x_;
    y(i+1) = recwindow(x_ - 0.5*t_s);
end

fig = figure,
subplot(2,1,1),plot(x,y,'LineWidth',1.5);     %stem(x,y,'filled');
title('Sequence in Time Domain','FontSize',12);
xlabel('Time');
ylabel('Amplitude');
y_f = fftshift(abs(fft(y(zero:zero+n),N+1)));
subplot(2,1,2),plot(2*(-N/2:N/2)/N,y_f,'LineWidth',1.5);   %stem(x,y_f,'filled');
title('Sequence in Frequency Domain','FontSize',12);
xlabel('x π rad/sample, Frequency');
ylabel('Amplitude');
title_ = ["T_s="+num2str(t_s)];
file_name = ["SamplingRecWin_Ts="+num2str(t_s)+"_2.bmp"];
sgtitle(title_);
saveas(fig,file_name);

% Task 03
[b,a] = butter(6,0.6);
y_1 = filter(b,a,y);
fig = figure,
subplot(2,1,1),plot(x,y_1,'LineWidth',1.5);     %stem(x,y,'filled');
title('Sequence in Time Domain','FontSize',12);
xlabel('Time');
ylabel('Amplitude');
y_f = fftshift(abs(fft(y_1(zero:zero+n),N+1)));
subplot(2,1,2),plot(2*(-N/2:N/2)/N,y_f,'LineWidth',1.5);   %stem(x,y_f,'filled');
title('Sequence in Frequency Domain','FontSize',12);
xlabel('x π rad/sample, Frequency');
ylabel('Amplitude');
title_ = ["T_s="+num2str(t_s)];
sgtitle(title_);
file_name = ["SamplingRecWin_Ts="+num2str(t_s)+"_3.bmp"];
saveas(fig,file_name);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [y]=recwindow( x )
    if (x<0) || (x>10)
        y = 0;
    else
        y = 1;
    end
end