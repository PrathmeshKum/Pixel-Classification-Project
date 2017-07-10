disp(' ## Program for Assignment 1');
disp(' Type "1" as model number for model based on rgb ');
disp(' Type "2" as model number for model based on hsv ');
disp(' Type "3" as model number for model based on YCbCr ');
disp(' Type "4" as model number for model based on combination of rgb & hsv ');
disp(' Type "5" as model number for model based on combination of rgb & YCbCr ');
disp(' Type "6" as model number for model based on combination of rgb, hsv & YCbCr ');
disp(' Type "7" as model number for model based on combination of hsv & grayscale ');
directory=char(pwd);

number = input(' Enter the model number to use: ','s');

 if strcmp(number,'1')==1;
    
      rgb();
      path=[directory '\rgb_values.mat'];
      load(path);

  elseif strcmp(number,'2')==1;
    
        hsv();
        path=[directory '\hsv_values.mat'];
        load(path);
    
  elseif strcmp(number,'3')==1;
    
        ycbcr();
        path=[directory '\ycbcr_values.mat'];
        load(path);
 
  elseif strcmp(number,'4')==1;
    
        rgb_hsv();
        path=[directory '\rgb_hsv_values.mat'];
        load(path);
  
  elseif strcmp(number,'5')==1;
    
        rgb_ycbcr();
        path=[directory '\rgb_ycbcr_values.mat'];
        load(path);      
        
  elseif strcmp(number,'6')==1;
    
        rgb_hsv_ycbcr();
        path=[directory '\rgb_hsv_ycbcr_values.mat'];
        load(path);
        
  elseif strcmp(number,'7')==1;
    
        hsv_gray();
        path=[directory '\hsv_gray_values.mat'];
        load(path);
        
    
 end
     
  