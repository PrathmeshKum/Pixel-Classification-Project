function [] = rgb_ycbcr()

clc;
clear all;
close all;
warning('off', 'Images:initSize:adjustingMag');

%file directory

directory=char(pwd);

trainingImageDirectory = [directory '\trainingImages\'];
testingImageDirectory = [directory '\testingImages\'];
annotatedTrainingImageDirectory = [directory '\annotatedTrainingImages_Wenjin\'];
annotatedTestingImageDirectory = [directory '\annotatedTestingImages_Wenjin\'];

%your parameters

nDim = 16; %number of bins

%training process

tt= cputime;
Pr_x_given_w_equalsTo_1 = zeros(nDim,nDim,nDim,nDim,nDim,nDim);
Pr_x_given_w_equalsTo_0 = zeros(nDim,nDim,nDim,nDim,nDim,nDim);    
trainingImageFiles = dir(trainingImageDirectory);
annotatedTrainingImageFiles = dir(annotatedTrainingImageDirectory);
N_face=0;

 for iFile = 3:size(trainingImageFiles,1);  
     
        %load the image and facial image regions
        
        origIm=imread([trainingImageDirectory trainingImageFiles(iFile).name]);    
        bwMask = imread([annotatedTrainingImageDirectory annotatedTrainingImageFiles(iFile).name]);    
        
        %generating the mask indicating the facial regions
        
        origIm = imresize(origIm,0.125*0.5);
        bwMask = imresize(bwMask,0.125*0.5);
        [nrows,ncols,~]= size(origIm);
        
        %your code to compute prior
        
        N_face = N_face + sum(bwMask(:));


        %your code to compute likelihood
        
        origIm1=rgb2ycbcr(origIm); % converts rgb values into ycbcr in range (y: 16-235, cb & cr: 16-240)
        
        % visualisation
        
        figure;
        subplot(2,1,1);
        showIm = origIm; showIm(bwMask) = 255;
        imshow(showIm,[]);
        subplot(2,1,2);
        showIm = origIm1; showIm(bwMask) = 255;
        imshow(showIm,[]);
        
        for iRow = 1:nrows;
            
            for iCol = 1:ncols;
                
                Y=origIm1(iRow,iCol,1);
                CB=origIm1(iRow,iCol,2);
                CR=origIm1(iRow,iCol,3);
                
                y=(floor((Y-16)*0.1435*0.45))+1; % for converting into range: 1-8
                cb=(floor((CB-16)*0.1420*0.45))+1;
                cr=(floor((CR-16)*0.1420*0.45))+1; 
                
                r=origIm(iRow,iCol,1);
                r=(floor(r*0.1216*0.45))+1; % for converting to range: 1-8
                g=origIm(iRow,iCol,2);
                g=(floor(g*0.1216*0.45))+1;
                b=origIm(iRow,iCol,3);
                b=(floor(b*0.1216*0.45))+1;
      
                
                if bwMask(iRow,iCol)==1;
                    
                    Pr_x_given_w_equalsTo_1(r,g,b,y,cb,cr) = Pr_x_given_w_equalsTo_1(r,g,b,y,cb,cr) + 1;
                
                else
                    
                    Pr_x_given_w_equalsTo_0(r,g,b,y,cb,cr) = Pr_x_given_w_equalsTo_0(r,g,b,y,cb,cr) + 1;
               
                end
            end
        end
        
 end
 
 % for computation of normalized likehood
 
  Pr_x_given_w_equalsTo_1 = Pr_x_given_w_equalsTo_1/(sum(Pr_x_given_w_equalsTo_1(:)));
  Pr_x_given_w_equalsTo_0 = Pr_x_given_w_equalsTo_0/(sum(Pr_x_given_w_equalsTo_0(:)));
 
 % for computation of prior
 
 nfiles=length(3:size(trainingImageFiles,1));
 
 Pr_w_equalsTo_1 = (N_face)/(nfiles*nrows*ncols);
 Pr_w_equalsTo_0 = 1 - (Pr_w_equalsTo_1);
 
 %your code to make the prior and likehood as distributions
  
 
 disp(['traning: ' num2str(cputime-tt)]);
 
 %testing
 
testingFiles = dir(testingImageDirectory);
annotatedTestingImageFiles = dir(annotatedTestingImageDirectory);
file_num=1;

for iFile = 3:size(testingFiles,1)
    tt = cputime;
    
    %load the image and facial image regions
    
    origIm=imread([testingImageDirectory testingFiles(iFile).name]);    
    %detMask = imread([annotatedTestingImageDirectory annotatedTestingImageFiles(iFile).name]);
    
    origIm = imresize(origIm,0.125*0.5);
    [nrows, ncols,~] = size(origIm);
    detMask=zeros(nrows,ncols);
    
    %your code to do the inference
    
    origIm2=rgb2ycbcr(origIm); % converts rgb values into ycbcr in range (y: 16-235, cb & cr: 16-240)   
   
    
     for iRow = 1:nrows;
            
            for iCol = 1:ncols;
                
                Y=origIm2(iRow,iCol,1);
                CB=origIm2(iRow,iCol,2);
                CR=origIm2(iRow,iCol,3);
                
                y=(floor((Y-16)*0.1435*0.45))+1; % for converting into range: 1-8
                cb=(floor((CB-16)*0.1420*0.45))+1;
                cr=(floor((CR-16)*0.1420*0.45))+1; 
                
                r=origIm(iRow,iCol,1);
                r=(floor(r*0.1216*0.45))+1; % for converting to range: 1-8
                g=origIm(iRow,iCol,2);
                g=(floor(g*0.1216*0.45))+1;
                b=origIm(iRow,iCol,3);
                b=(floor(b*0.1216*0.45))+1;
                
                if (Pr_x_given_w_equalsTo_1(r,g,b,y,cb,cr)*Pr_w_equalsTo_1) > (Pr_x_given_w_equalsTo_0(r,g,b,y,cb,cr)*Pr_w_equalsTo_0);
                    
                    detMask(iRow,iCol)=1;
                
                else
                    
                    continue;
               
                end
            end
      end
    
    

    %your code to compute the TP, FP,FN
    
    % for computation of ground truth (gtMask)
    
    gtMask = imread([annotatedTestingImageDirectory annotatedTestingImageFiles(iFile).name]);
    gtMask = imresize(gtMask,0.125*0.5);
    
    % True Positive (TP), False Positive (FP), False Negative (FN)
    
    detMask1=detMask;
    gtMask1=gtMask;
    tp=zeros(nrows,ncols);
    
    for iRow = 1:nrows;
        
        for iCol = 1:ncols;
            
            if detMask1(iRow,iCol)==1 && gtMask1(iRow,iCol)==1;
                
                tp(iRow,iCol)=1;
            
            else
                
                continue
            
            end
        end    
    end        
     
    TP(1,file_num)=(sum(sum(tp)));
    
    for iRow = 1:nrows;
        
        for iCol = 1:ncols;
            
            if detMask1(iRow,iCol)==1 && tp(iRow,iCol)==1;
                
                detMask1(iRow,iCol)=0;
            
            else
                
                continue
            
            end
        end    
    end
    
    fp=(sum(sum(detMask1)));
    FP(1,file_num)=fp;
    
    for iRow = 1:nrows;
        
        for iCol = 1:ncols;
            
            if gtMask1(iRow,iCol)==1 && tp(iRow,iCol)==1;
                
                gtMask1(iRow,iCol)=0;
            
            else
                
                continue
            
            end
        end    
    end
    
    fn=(sum(sum(gtMask1)));
    FN(1,file_num)=fn;
    
    disp([num2str(iFile-2) ' testing: ' num2str(cputime-tt)]);
    
    %some visualization
    
    figure;
    subplot(2,1,1);
    showIm = origIm; showIm(nrows*ncols+find(detMask)) = 255;
    imshow([origIm repmat(255*detMask,[1 1 3]) showIm],[]);
    subplot(2,1,2);
    showIm = origIm2; showIm(nrows*ncols+find(detMask)) = 255;
    imshow([origIm2 repmat(255*detMask,[1 1 3]) showIm],[]);
 
    file_num=file_num+1;
    
end
    
%your code to compute the precision, recall and F-score
 
 precision=((sum(TP)/file_num)/((sum(TP)/file_num)+(sum(FP)/file_num)))*100; % In Percentage
 recall=((sum(TP)/file_num)/((sum(TP)/file_num)+(sum(FN)/file_num)))*100; % In Percentage
 f_score=((2*precision*recall)/(precision+recall));
path=[directory '\rgb_ycbcr_values.mat'];
save(path);