clear;clc;
load('S.mat');

for choose_num=1:7
    for rand_num=1:10
        L={}; L_test={}; len=[]; indd=[];
        for j=1:5
            i=choose_num;k1=1;k2=1;
            k=ceil(0.7*S(i).file(j).num);
            indd=randperm(S(i).file(j).num);
            for l=1:S(i).file(j).num
                len(l,j)=length(S(i).file(j).data(l).d(:,1));
                start_index=S(i).file(j).data(l).start_index;
                switch j
                    case {1,2}
                        s=start_index+2250;
                    case {3,4}
                        s=start_index+3000;
                    case {5}
                        s=start_index+1500;
                end
                if ismember(l,indd(1:k))
                    L{k1,j}=S(i).file(j).data(l).d(start_index:s,2:13);
                    k1=k1+1;
                else
                    L_test{k2,j}=S(i).file(j).data(l).d(start_index:s,2:13);
                    k2=k2+1;
                end
            end
        end
        
        Fet_1=[];Lab_1=[];x=[];Fs_1={}; Fet_test=[];Lab_2=[];Fs_2={};
        d0=1;u=1; d1=1;u1=1;
        for j=1:5
            for k=1:size(L,1)
                if isempty(L{k,j})
                else
                    m0=fix((length(L{k,j})-300)/100)+1;
                    for b=1:m0
                        x=L{k,j}(1+100*(b-1):300+100*(b-1),:);
                        Fs_1{u,1}=x;
                        Fet_1(d0,:)=feture_calc(x,1500);
                        switch j
                            case 1
                                Fs_1{u,2}=1;
                                Lab_1(d0,1)=1;
                            case 2
                                Fs_1{u,2}=2;
                                Lab_1(d0,1)=2;
                            case {3,4}
                                Fs_1{u,2}=3;
                                Lab_1(d0,1)=3;
                            case 5
                                Fs_1{u,2}=4;
                                Lab_1(d0,1)=4;
                        end
                        d0=d0+1;
                        u=u+1;
                    end
                end
            end
        end
        
        for j=1:5
            for k=1:size(L_test,1)
                if isempty(L_test{k,j})
                else
                    m0=fix((length(L_test{k,j})-300)/100)+1;
                    for b=1:m0
                        x=L_test{k,j}(1+100*(b-1):300+100*(b-1),:);
                        Fs_2{u1,1}=x;
                        Fet_test(d1,:)=feture_calc(x,1500);
                        switch j
                            case 1
                                Fs_2{u1,2}=1;
                                Lab_2(d1,1)=1;
                            case 2
                                Fs_2{u1,2}=2;
                                Lab_2(d1,1)=2;
                            case {3,4}
                                Fs_2{u1,2}=3;
                                Lab_2(d1,1)=3;
                            case 5
                                Fs_2{u1,2}=4;
                                Lab_2(d1,1)=4;
                        end
                        d1=d1+1;
                        u1=u1+1;
                        %                 end
                    end
                end
            end
        end
        
        Fet=[];Fet_pre=[];
        [~,r2] = size(Fet_1);
        for i=1:r2
            Fet(:,i)=(Fet_1(:,i)-min(Fet_1(:,i)))/(max(Fet_1(:,i))-min(Fet_1(:,i)));
        end
        [~,r2] = size(Fet_test);
        for i=1:r2
            Fet_pre(:,i)=(Fet_test(:,i)-min(Fet_test(:,i)))/(max(Fet_test(:,i))-min(Fet_test(:,i)));
        end
        
        randIndex=[];Fet_tran=[];Lab_tran=[];Fet_vaild=[];Lab_vaild=[];
        randIndex = randperm(length(Fet));
        a=length(Fet);
        Fet_tran=Fet(randIndex(1:a),:);
        Lab_tran=Lab_1(randIndex(1:a),:);
        Fet_vaild=Fet_pre;
        Lab_vaild=Lab_2;
        
        sample_x=[];sample_d=[];
        sample_x=sort(Lab_tran);
        sample_d=diff([sample_x;max(sample_x)+1]);
        count = diff(find([1;sample_d])) ;
        
        train_data=[];test_data=[];
        train_data=[Fet_tran,Lab_tran];
        test_data=[Fet_vaild,Lab_vaild];
        
        LDA_trainedClassifier=[];FLDA_trainedClassifier=[];DT_trainedClassifier=[];BES_trainedClassifier=[];
        KNN_trainedClassifier=[];LSVM_trainedClassifier=[];FSVM_trainedClassifier=[];RBFSVM_trainedClassifier=[];
        [LDA_trainedClassifier, LDA_validationAccuracy] = LDA(train_data);
        [FLDA_trainedClassifier, FLDA_validationAccuracy] = FLDA(train_data);
        [DT_trainedClassifier, DT_validationAccuracy] = DT(train_data);
        [BES_trainedClassifier, BES_validationAccuracy] = BES(train_data);
        [KNN_trainedClassifier, KNN_validationAccuracy] = KNN(train_data);
        [LSVM_trainedClassifier, LSVM_validationAccuracy] = LSVM(train_data);
        [FSVM_trainedClassifier, FSVM_validationAccuracy] = FSVM(train_data);
        [RBFSVM_trainedClassifier, RBFSVM_validationAccuracy] = RBFSVM(train_data);
        
        pred_LDA=[];pred_FLDA=[];pred_DT=[];pred_BES=[];pred_KNN=[];pred_LSVM=[];pred_FSVM=[];pred_RBFSVM=[];
        pred_LDA = LDA_trainedClassifier.predictFcn(Fet_vaild);
        pred_FLDA = FLDA_trainedClassifier.predictFcn(Fet_vaild);
        pred_DT = DT_trainedClassifier.predictFcn(Fet_vaild);
        pred_BES = BES_trainedClassifier.predictFcn(Fet_vaild);
        pred_KNN = KNN_trainedClassifier.predictFcn(Fet_vaild);
        pred_LSVM = LSVM_trainedClassifier.predictFcn(Fet_vaild);
        pred_FSVM = FSVM_trainedClassifier.predictFcn(Fet_vaild);
        pred_RBFSVM = RBFSVM_trainedClassifier.predictFcn(Fet_vaild);
        
        acc_LDA = sum(pred_LDA == Lab_vaild)./length(pred_LDA);
        acc_FLDA = sum(pred_FLDA == Lab_vaild)./length(pred_FLDA);
        acc_DT = sum(pred_DT == Lab_vaild)./length(pred_DT);
        acc_BES = sum(pred_BES == Lab_vaild)./length(pred_BES);
        acc_KNN = sum(pred_KNN == Lab_vaild)./length(pred_KNN);
        acc_LSVM = sum(pred_LSVM == Lab_vaild)./length(pred_LSVM);
        acc_FSVM = sum(pred_FSVM == Lab_vaild)./length(pred_FSVM);
        acc_RBFSVM = sum(pred_RBFSVM == Lab_vaild)./length(pred_RBFSVM);

        
        Fet_ANNtran=[];Lab_ANNtran=[];
        Fet_ANNtran=Fet_tran';
        for i=1:size(Fet_ANNtran,2)
            switch Lab_tran(i)
                case 1
                    Lab_ANNtran(:,i)= [1 0 0 0]';
                case 2
                    Lab_ANNtran(:,i)= [0 1 0 0]';
                case 3
                    Lab_ANNtran(:,i)= [0 0 1 0]';
                case 4
                    Lab_ANNtran(:,i)= [0 0 0 1]';
            end
        end
        Fet_ANNtest=Fet_vaild';
        for i=1:size(Fet_ANNtest,2)
            switch Lab_vaild(i)
                case 1
                    LAB_ANNtest(:,i)= [1 0 0 0]';
                case 2
                    LAB_ANNtest(:,i)= [0 1 0 0]';
                case 3
                    LAB_ANNtest(:,i)= [0 0 1 0]';
                case 4
                    LAB_ANNtest(:,i)= [0 0 0 1]';
            end
        end
        
        pred_ANN=[];
        pred_ANN = ANN(Fet_ANNtran,Lab_ANNtran,Fet_ANNtest);
        coun_ANN=0;
        for L_ANN=1:length(pred_ANN)
            [~,index]=max(pred_ANN(:,L_ANN));
            switch index
                case 1
                    pred_ANN(:,L_ANN)= [1 0 0 0]';
                case 2
                    pred_ANN(:,L_ANN)= [0 1 0 0]';
                case 3
                    pred_ANN(:,L_ANN)= [0 0 1 0]';
                case 4
                    pred_ANN(:,L_ANN)= [0 0 0 1]';
            end
            if pred_ANN(:,L_ANN) == LAB_ANNtest(:,L_ANN)
                coun_ANN=coun_ANN+1;
            end
        end
        
        acc_ANN = coun_ANN/length(pred_ANN);
        ACC(choose_num).acc(rand_num,:)=[acc_DT,acc_LDA,acc_FLDA,acc_BES,acc_LSVM,acc_FSVM,acc_RBFSVM,acc_KNN,acc_ANN];
    end
end



















