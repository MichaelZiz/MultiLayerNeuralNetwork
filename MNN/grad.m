function [ grads] = grad( y,z )
%function implement J(w)
    grads =zeros(4,1);
     
    for i=1:4
        
   
        grads(i,1)=grads(i,1) + (y(i,1)-z(i,1))^2;
        
    end
    
    
    grads=(0.5)*grads;
    
   
    
    
    
    
end

