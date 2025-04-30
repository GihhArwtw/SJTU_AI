audio_file_path = "./宇多田ヒカル - 夕凪.mp3";
audio_name = "夕凪";


[y,Fs] = audioread(audio_file_path);
duration = length(y);

% TASK 04

% Some special filters.
% The attenuation of each filter in the stopband.
Rs_v = 30;
Rs_p = 30;

% Show the frequency response of each filter.
[n,Wn] = Vocal_Enhance( Rs_v, Fs );
fig = Display( n,Wn,900,300,Fs,"Filter for Vocal Enhancement.","vocal.bmp");
[n,Wn] = Percussion_Enhance( Rs_p, Fs );
fig = Display( n,Wn,900,300,Fs,"Filter for Percussion Enhancement.", "perc.bmp");


% The parameters in the equalizer are the scale of enhancement or 
%     attenuation (negative number) of different components.
% The commended range of parameters are [-2,2].

file_name = [audio_name + "_Vocal.wav"];
z1 = Combination(y,Fs,Rs_v,Rs_p,1,0,file_name);

file_name = [audio_name + "_Perc.wav"];
z2 = Combination(y,Fs,Rs_v,Rs_p,0,2,file_name);

file_name = [audio_name + "_edited.wav"];
z_1 = Combination(y,Fs,Rs_v,Rs_p,1,2,file_name);



% Equalizer Design.
% The attenuation of each filter in the stopband.
Rs = 30;

% Show the frequency response of the set of filters.
fig = figure("Position",[0.1,0.1,900,500]);
subplot(2,1,1);
freq = [0,30,63,125,250,500,1000,2000,4000,8000,16000];
for i = 1:10
   b = (freq(i)*2+freq(i+1))/3;
   c = (freq(i)+freq(i+1)*2)/3;
   [n,Wn] = Band_Pass(0.5*freq(i),b,c,2*freq(i+1),Rs,Fs);
   [z,p] = butter(n,Wn);
   [h,f] = freqz(z,p,512,Fs);
   plot(f, abs(h),"LineWidth",1.5);
   hold on;
end
xlabel("Frequency (Hz)","FontName","serif","FontSize",12);
ylabel("Gain","FontName","serif","FontSize",12);

subplot(2,1,2);
for i = 1:10
    b = (freq(i)*2+freq(i+1))/3;
    c = (freq(i)+freq(i+1)*2)/3;
    [n,Wn] = Band_Pass(0.5*freq(i),b,c,2*freq(i+1),Rs,Fs);
    [z,p] = butter(n,Wn);
    n = 512;
    if (freq(i)*20<Fs)
        n = 4096;
    end
    if (freq(i)*500<Fs)
        n = 32768;
    end
    [h,f] = freqz(z,p,n,Fs);
    semilogx(f, abs(h),"LineWidth",1.5);
    hold on;
end
xlabel("Frequency (Hz)","FontName","serif","FontSize",12);
ylabel("Gain","FontName","serif","FontSize",12);
sgtitle("The Frequency Respoense of Equalizer Filters","FontName","serif","FontSize",15);
saveas(fig,"equalizer.bmp");


a = [1.0, 0.8, 0.5, 0.3, -1, -3, 2, -1, -2, 3];

file_name = [audio_name + "_equalizer.wav"];
z = Equalizer(y,Fs,Rs,a,file_name);



%%%%%%%%%%%%%%%%%%%
function [n,Wn] = Vocal_Enhance( Rs, Fs )
    Wp = [3000,3900]/Fs;
    Ws = [300,5000]/Fs;
    [n,Wn] = buttord(Wp,Ws,3,Rs);
end

%%%%%%%%%%%%%%%%%%%
function [n,Wn] = Percussion_Enhance( Rs, Fs )
    Wp = 100/Fs;
    Ws = 500/Fs;
    [n,Wn] = buttord(Wp,Ws,3,Rs);
end

%%%%%%%%%%%%%%%%%%%
function [n,Wn] = Band_Pass( a,b,c,d,Rs,Fs )
    if (a>0)
        Wp = [b,c]/Fs;
    else
        Wp = c/Fs;
    end
    
    if (a>0)
        Ws = [a,d]/Fs;
    else
        Ws = d/Fs;
    end
    [n,Wn] = buttord(Wp,Ws,3,Rs);
end

%%%%%%%%%%%%%%%%%%%
function [fig] = Display( n, Wn, scale_x, scale_y, Fs, title_, file_name )
    fig = figure("Position",[0.1,0.1,scale_x,scale_y]);
    [z,p] = butter(n,Wn);
    [h,f] = freqz(z,p,512,Fs);
    plot(f, abs(h),"LineWidth",1.5);
    title(title_,"FontName","serif","FontSize",15);
    xlabel("Frequency (Hz)","FontName","serif","FontSize",12);
    ylabel("Gain","FontName","serif","FontSize",12);
    saveas(fig,file_name);
end

%%%%%%%%%%%%%%%%%%%%
function [z] = Combination( y, Fs, Rs_v, Rs_p, Vocal, Percussion, file_name )
    [n,Wn] = Vocal_Enhance( Rs_v, Fs );
    [b,a] = butter(n,Wn);
    y_v = filter(b,a,y);

    [n,Wn] = Percussion_Enhance( Rs_p, Fs );
    [b,a] = butter(n,Wn);
    y_p = filter(b,a,y);

    for i = 1:length(y(1,:))
        y_v(:,i) = y_v(:,i)/max(y_v(:,i));
        y_p(:,i) = y_p(:,i)/max(y_p(:,i));
    end

    z =  y + Vocal*y_v + Percussion*y_p;
    audiowrite(file_name,z,Fs);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [z] = Equalizer( y, Fs, Rs, a, file_name )
    freq = [0,30,63,125,250,500,1000,2000,4000,8000,16000];
    for i = 1:10
        b = (freq(i)*2+freq(i+1))/3;
        c = (freq(i)+freq(i+1)*2)/3;
        [n,Wn] = Band_Pass(0.5*freq(i),b,c,2*freq(i+1),Rs,Fs);
        [k,p] = butter(n,Wn);
        y_n = filter(k,p,y);
        z = y + y_n*a(i);
    end

    audiowrite(file_name,z,Fs);
end