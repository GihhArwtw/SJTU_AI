MAX = 7;            % if set MAX = 8, len = 4294967296, x will cause a memory overflow.
time_ = zeros(4,MAX);
single_time_ = zeros(4,MAX);
COUNT = 10;

for count = 1:COUNT
    for i = 1:MAX
        profile on
        len = 2^(4*i);
        x = randi([-5.,5.],len,1);
        len
        y = our_DFT(x);
        if (y==-1)
            single_time_(1,i) = -1;
        end
        y = our_Matrix_DFT(x);
        if (y==-1)
            single_time_(2,i) = -1;
        end
        y = in_FFT(x);
        y = gpu_FFT(x);
        if (y==-1)
            single_time_(4,i) = -1;
        end
    
        p = profile('info');
        for j = 1:4
            if (single_time_(j,i)>-1)
                single_time_(j,i) = p.FunctionTable(j+1).TotalTime;
            else
                single_time_(j,i) = nan;
            end
        end
        profile off

        for j = 1:4
            time_(j,i) = time_(j,i) + single_time_(j,i);
        end
    end
end

for i = 1:MAX
    for j = 1:4
        time_(j,i) = time_(j,i) / COUNT;
    end
end

cString = {'#008e61','#663275', '#c75769', '#637ca1'}

fig = figure, subplot(2,1,1),
x = linspace(1,MAX,MAX);
z = [];
for i = 1:4
    for j = 1:MAX
        z(j) = time_(i,j);
    end
    semilogy(x,z,'-*','color',cString{i},'LineWidth',1);
    hold on
end
legend('For-cyles','Matrix','FFT','FFT_{GPU}','Location','southeast');
xticks(x);
xticklabels({'16','256','4096','65536','2^{20}','2^{24}','2^{28}'});
xlabel('Length');
ylabel('Time Consumption (s)');
set(fig,'position',[0.1,0.1,800,500]); 
title('Comparison of Computation Time of Different Ways Calculating DFT','FontSize',12);

subplot(2,1,2),
MAX = 5;
x = x(1:MAX);
z = [];
for i = 1:4
    for j = 1:MAX
        z(j) = time_(i,j);
    end
    semilogy(x,z,'-*','color',cString{i},'LineWidth',1);
    hold on
end
legend('For-cyles','Matrix','FFT','FFT_{GPU}','Location','northeast');
xticks(x);
xticklabels({'16','256','4096','65536','2^{20}'});
xlabel('Length');
ylabel('Time Consumption (s)');
set(fig,'position',[0.1,0.1,800,675]); 
title('Comparison of Computation Time When Length<2^{24}','FontSize',12);


saveas(fig,"Results.svg");




%%%%%%%%%%%%%%%%%%%%%%%

function [X] = our_DFT( x )
    N = length(x);
    if N>=65536              % >= 65536 will cause memory overflow.
        X = -1;          
    else
        X = zeros(N);
        for i = 1:N
            for j = 1:N
                X(i) = X(i) + x(j) * exp(-2j*pi*j/N);
            end
        end
    end
end

function [X] = our_Matrix_DFT( x )
    N = length(x);
    if N>=65536            % >= 65536 will cause memory overflow.
        X = -1;
    else
        W = ones(N,N);
        for i = 2:N
            for j = 2:N
                W(i,j) = exp(-2j*pi*(i-1)*(j-1)/N);
            end
        end
        X = N*x;
    end
end

function [X] = in_FFT( x )
    X = fft(x);
end

function [X] = gpu_FFT( x )
    if (length(x)>=268435456)
        X = -1;
    else
        X = fft(gpuArray(x));
    end
end