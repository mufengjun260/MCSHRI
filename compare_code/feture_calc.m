function [F] = feture_calc(A,s)

f1=[]; f2=[]; f3=[]; f4=[]; f5=[]; F=[];
[r1,r2]=size(A);

for i=1:r2
    f1(i)=mean(abs(A(:,i)));
end

for i=1:r2
    wl=0;
    for j=1:r1-1
        wl=wl+abs(A(j+1,i)-A(j,i));
    end
    f2(i)=wl/r1;
end

for i=1:r2
    count=0;
    for j=1:r1-1
        if A(j+1,i)*A(j,i) <= 0
            count=count+1;
        else
            count=count;
        end
    end
    f3(i)=count;
end

%AR自回归系数
for i=1:r2
    c=[];
    c = aryule(A(:,i),6);
    f4(i)=c(2);
end

%功率谱密度
for i=1:r2
    [psdestx,~] = pwelch(A(:,i),hamming(r1),0,r1,s);
    f5(i)=mean(psdestx);
end
j=1;
for i=1:5:5*r2
    F(i)=f1(j);
    F(i+1)=f2(j);
    F(i+2)=f3(j);
    F(i+3)=f4(j);
    F(i+4)=f5(j);
    j=j+1;
end



