function[beta,sparsity]=SQR(Y,X,Q,C,s)
    
    iteration=10;
    S=size(X);
    b=C*log(S(2))/S(2);
    a=0.5;
    C1=(S(2)+a/(2.^s));
    power=1/(2-0.5.^s);
    beta=zeros(S(2),1);
    Z=ones(S(2),1);

    for t=2:iteration

        L1_error=max(abs(Y-X*beta),0.1);
        T=Y-(1-2*Q)*L1_error;

        W=0.25*spdiags(L1_error.^-1,0,S(1),S(1));

        XTW=X'*W;

        XTWX=sum(XTW.*X',2);

        iter=1;

        ink1=X(:,2:end)*beta(2:end,:);
        ink2=zeros(S(1),1);
        beta_previous=ones(S(2),1);

        while norm(beta-beta_previous)>1e-4 && iter<20

            beta_previous=beta;
            iter=iter+1;

            for j=1:S(2)

                C2=1/b+sum(abs(beta).^(0.5.^s))-abs(beta(j)).^(0.5.^s);

                if (j~=1) && (j~=S(2))
                    Z(j)=XTW(j,:)*(T-ink1-ink2)/XTWX(j);
                elseif j==1
                    Z(1)=XTW(1,:)*(T-ink1)/XTWX(j);
                else
                    Z(S(2))=XTW(S(2),:)*(T-ink2)/XTWX(j);
                end

                if abs(Z(j))<=2*(C1/(2*C2+2*abs(Z(j)).^0.5)/XTWX(j)).^power

                    beta(j)=0;

                else

                    beta_old=abs(Z(j));
                    beta_new=10;
                    k=1;

                    while abs(beta_old-beta_new)>1e-4 && k<20 && beta_new>=0
                        
                        beta_old=beta_new;
                        beta_new=abs(Z(j))-C1/XTWX(j)/(beta_old+C2*beta_old.^(1-0.5.^s));
                        k=k+1;
                    end
                        
                    if k>=20 || beta_new<0
                        beta(j)=0; 
                    else
                        beta(j)=beta_new*sign(Z(j));
                    end

                end

                if (j~=1) && (j~=S(2))
                    ink1=ink1-X(:,j+1)*beta(j+1,:);
                    ink2=ink2+X(:,j)*beta(j,:);
                elseif j==1
                    ink1=ink1-X(:,2)*beta(2,:);
                    ink2=X(:,1)*beta(1,:);
                else
                    ink1=ink2+X(:,j)*beta(j,:)-X(:,1)*beta(1,:);
                    ink2=zeros(S(1),1);
                end

            end

        end

    end

    sparsity=nnz(beta);

end