clearvars;
close all;
clc

params.kx=0; 
params.kz=1;
params.omega=0;
Ny=256;
params.Ny=Ny;
params.Re=358;
params.Ly=2;
params.gmres_tol=1e-6;
% Preconditioner options:
%   'none'                 : no GMRES preconditioner
%   'fast_scalar_laplacian': LU of scalar Chebyshev Laplacian from chebdif
%   'green_laplacian'      : matrix-free Green's function inverse by cumulative sums
params.preconditioner_type='fast_scalar_laplacian';
params=prepare_params(params);

benchmark_Ny = Ny;

[H,L]=resolvent_1D(params);
params = add_laplacian_preconditioner(params);

nsv = 1;

H_mf = @(f,tflag) H_fun(f,tflag,params);
[U_mf,S_mf,V_mf] = svds(H_mf,[3*Ny,3*Ny],nsv,'largest');
[U_full,S_full,V_full] = svds(H,nsv,'largest');

sigma_mf = diag(S_mf);
sigma_full = diag(S_full);
sigma_rel_err = abs(sigma_mf-sigma_full)./abs(sigma_full);

[U_mf_aligned,u_err] = align_singular_vectors(U_mf,U_full);
[V_mf_aligned,v_err] = align_singular_vectors(V_mf,V_full);

comparison_table = table(sigma_mf,sigma_full,sigma_rel_err,u_err.',v_err.', ...
    'VariableNames',{'sigma_matrix_free','sigma_full','sigma_relative_error', ...
    'left_vector_error','right_vector_error'});
disp(comparison_table)

figure;
tiledlayout(2,2)

nexttile
bar(categorical({'matrix-free','full matrix'}),[sigma_mf(1),sigma_full(1)])
ylabel('\sigma')
title('Largest singular value')

nexttile
semilogy(1:nsv,sigma_rel_err,'o-')
xlabel('singular value index')
ylabel('relative error')
title('Singular value error')
grid on

nexttile
plot(abs(U_full(:,1)),'k-',LineWidth=1.5)
hold on
plot(abs(U_mf_aligned(:,1)),'r--',LineWidth=1.2)
legend('full matrix','matrix-free','Location','best')
xlabel('state index')
ylabel('|u_1|')
title('Left singular vector')
grid on

nexttile
plot(abs(V_full(:,1)),'k-',LineWidth=1.5)
hold on
plot(abs(V_mf_aligned(:,1)),'r--',LineWidth=1.2)
legend('full matrix','matrix-free','Location','best')
xlabel('forcing index')
ylabel('|v_1|')
title('Right singular vector')
grid on

benchmark_table = benchmark_svd_methods(params,benchmark_Ny,nsv);
disp(benchmark_table)

figure;
tiledlayout(1,2)

nexttile
plot(benchmark_table.Ny,benchmark_table.full_total_time_s,'ko-',LineWidth=1.5)
hold on
plot(benchmark_table.Ny,benchmark_table.matrix_free_time_s,'rs--',LineWidth=1.5)
legend('full matrix','matrix-free','Location','best')
xlabel('Ny')
ylabel('time (s)')
title('SVD timing')
grid on

nexttile
plot(benchmark_table.Ny,benchmark_table.full_memory_MB,'ko-',LineWidth=1.5)
hold on
plot(benchmark_table.Ny,benchmark_table.matrix_free_memory_MB,'rs--',LineWidth=1.5)
legend('full matrix','matrix-free','Location','best')
xlabel('Ny')
ylabel('memory estimate (MB)')
title('Dominant array memory')
grid on


function benchmark_table = benchmark_svd_methods(params_template,Ny_list,nsv)
    nNy = numel(Ny_list);
    full_build_time_s = zeros(nNy,1);
    full_svd_time_s = zeros(nNy,1);
    full_total_time_s = zeros(nNy,1);
    matrix_free_time_s = zeros(nNy,1);
    sigma_full = zeros(nNy,1);
    sigma_matrix_free = zeros(nNy,1);
    full_memory_MB = zeros(nNy,1);
    matrix_free_memory_MB = zeros(nNy,1);

    for iNy = 1:nNy
        params_i = params_template;
        params_i.Ny = Ny_list(iNy);
        params_i = prepare_params(params_i);
        op_size = [3*params_i.Ny,3*params_i.Ny];

        drawnow;
        tic
        [H_i,L_i] = resolvent_1D(params_i);
        full_build_time_s(iNy) = toc;

        tic
        [U_full_i,S_full_i,V_full_i] = svds(H_i,nsv,'largest');
        full_svd_time_s(iNy) = toc;
        full_total_time_s(iNy) = full_build_time_s(iNy)+full_svd_time_s(iNy);
        sigma_full(iNy) = S_full_i(1,1);

        full_memory_MB(iNy) = workspace_bytes( ...
            H_i,L_i,U_full_i,S_full_i,V_full_i)/1024^2;

        params_i = add_laplacian_preconditioner(params_i);

        clear H_i U_full_i S_full_i V_full_i
        drawnow;

        H_mf_i = @(f,tflag) H_fun(f,tflag,params_i);
        tic
        [U_mf_i,S_mf_i,V_mf_i] = svds(H_mf_i,op_size,nsv,'largest');
        matrix_free_time_s(iNy) = toc;
        sigma_matrix_free(iNy) = S_mf_i(1,1);

        matrix_free_memory_MB(iNy) = (workspace_bytes( ...
            params_i,U_mf_i,S_mf_i,V_mf_i)+matrix_free_workspace_bytes(params_i))/1024^2;

        clear H_mf_i U_mf_i S_mf_i V_mf_i L_i
    end

    sigma_relative_error = abs(sigma_matrix_free-sigma_full)./abs(sigma_full);

    benchmark_table = table(Ny_list(:),full_build_time_s,full_svd_time_s, ...
        full_total_time_s,matrix_free_time_s,full_memory_MB, ...
        matrix_free_memory_MB,sigma_full,sigma_matrix_free, ...
        sigma_relative_error, ...
        'VariableNames',{'Ny','full_build_time_s','full_svd_time_s', ...
        'full_total_time_s','matrix_free_time_s','full_memory_MB', ...
        'matrix_free_memory_MB','sigma_full','sigma_matrix_free', ...
        'sigma_relative_error'});
end


function bytes = matrix_free_workspace_bytes(params)
    n = 4*params.Ny;
    restart = min(100,n);

    % Complex double Krylov basis used by GMRES, plus a few work vectors.
    bytes_per_complex_double = 16;
    bytes = bytes_per_complex_double*(n*(restart+1)+6*n);
    if strcmp(params.preconditioner_type,'fast_scalar_laplacian')
        bytes = bytes+params.laplacian_preconditioner_nnz*(16+8);
    end
end


function params=prepare_params(params)
    if ~isfield(params,'preconditioner_type')
        params.preconditioner_type='none';
    end
    Ny=params.Ny;
    [params.cheb_y,~]=chebdif(Ny,1);
    params.U0=1-params.cheb_y.^2;
    params.U1=-2*params.cheb_y;

    [~,w]=clencurt(Ny-1);
    params.w=w(:);
    params.w_all=[params.w;params.w;params.w];
end


function params = add_laplacian_preconditioner(params)
    params.preconditioner_type = validatestring(params.preconditioner_type, ...
        {'none','fast_scalar_laplacian','green_laplacian'});

    if strcmp(params.preconditioner_type,'none')
        return
    end

    if strcmp(params.preconditioner_type,'green_laplacian')
        [y_sorted,sort_idx] = sort(params.cheb_y(:));
        unsort_idx = zeros(size(sort_idx));
        unsort_idx(sort_idx) = 1:numel(sort_idx);

        params.green_y = params.Ly/2*y_sorted;
        params.green_w = params.Ly/2*params.w(sort_idx);
        params.green_sort_idx = sort_idx;
        params.green_unsort_idx = unsort_idx;
        params.green_alpha = sqrt(params.kx^2+params.kz^2);
        return
    end

    Ny=params.Ny;
    [~,DM]=chebdif(Ny,2);
    D2=(2/params.Ly)^2*DM(:,:,2);
    lap=D2-(params.kx^2+params.kz^2)*eye(Ny);
    lap(1,:)=0; lap(1,1)=1;
    lap(Ny,:)=0; lap(Ny,Ny)=1;
    params.laplacian_factor=decomposition(lap,'lu');
    params.laplacian_preconditioner_nnz=nnz(lap);
end


function bytes = workspace_bytes(varargin)
    bytes = 0;
    for k = 1:nargin
        value = varargin{k}; %#ok<NASGU>
        info = whos('value');
        bytes = bytes+info.bytes;
    end
end


function [U_aligned,err] = align_singular_vectors(U,U_ref)
    U_aligned = U;
    err = zeros(1,size(U,2));
    for j = 1:size(U,2)
        phase = U_ref(:,j)'*U(:,j);
        if phase ~= 0
            U_aligned(:,j) = U(:,j)*conj(phase)/abs(phase);
        end
        err(j) = norm(U_aligned(:,j)-U_ref(:,j))/norm(U_ref(:,j));
    end
end


function H_f=H_fun(f,tflag,params)

    %kx,kz,omega,Ny,Re,Ly
    Ny=params.Ny;
    w_all=params.w_all;

    %if strcmp(tflag,'notransp')
        Bf=[f.*w_all.^(-1/2);
            zeros(Ny,1)];
    % else
    %     Bf=[f.*w_all.^(1/2);
    %         zeros(Ny,1)];
    % end
    
    %B.C. for u
    Bf(1)=0;
    Bf(Ny)=0;
    
    %B.C. for v
    Bf(Ny+1)=0;
    Bf(2*Ny)=0;

    %B.C. for w
    Bf(2*Ny+1)=0;
    Bf(3*Ny)=0;

    
    tol = params.gmres_tol;
    restart = min(100,4*Ny);
    maxit = 4*Ny;


    if isfield(params,'preconditioner_type') && ~strcmp(params.preconditioner_type,'none')
        [L_inv_u_p,flag,relres,iter] = gmres(@(u_p) L_fun(u_p,tflag,params),Bf, ...
            restart,tol,maxit,@(rhs) laplacian_preconditioner_fun(rhs,params));
    else
        [L_inv_u_p,flag,relres,iter] = gmres(@(u_p) L_fun(u_p,tflag,params),Bf, ...
            restart,tol,maxit);
    end
    if flag ~= 0
        if isscalar(iter)
            iter_msg = sprintf('%g',iter);
        else
            iter_msg = sprintf('[%d %d]',iter(1),iter(2));
        end
        warning('main_1D_resolvent:linearSolveNotConverged', ...
            'GMRES did not converge for %s solve. flag=%d, relres=%g, iter=%s.', ...
            tflag,flag,relres,iter_msg);
    end
    %if strcmp(tflag,'notransp')
    
        H_f = w_all.^(1/2).*L_inv_u_p(1:3*Ny,1);
    % else
    %     H_f = w_all.^(-1/2).*L_inv_u_p(1:3*Ny,1);
    % 
    % end
end


function z=laplacian_preconditioner_fun(rhs,params)
    Ny=params.Ny;

    z=zeros(4*Ny,1);
    if strcmp(params.preconditioner_type,'fast_scalar_laplacian')
        z(1:Ny)=params.laplacian_factor\rhs(1:Ny);
        z(Ny+1:2*Ny)=params.laplacian_factor\rhs(Ny+1:2*Ny);
        z(2*Ny+1:3*Ny)=params.laplacian_factor\rhs(2*Ny+1:3*Ny);
    elseif strcmp(params.preconditioner_type,'green_laplacian')
        z(1:Ny)=green_laplacian_inverse(rhs(1:Ny),params);
        z(Ny+1:2*Ny)=green_laplacian_inverse(rhs(Ny+1:2*Ny),params);
        z(2*Ny+1:3*Ny)=green_laplacian_inverse(rhs(2*Ny+1:3*Ny),params);
    else
        error('Unknown preconditioner type "%s".',params.preconditioner_type)
    end
    z(3*Ny+1:4*Ny)=rhs(3*Ny+1:4*Ny);
end


function u=green_laplacian_inverse(r,params)
    y=params.green_y;
    w=params.green_w;
    alpha=params.green_alpha;

    r_sorted = r(params.green_sort_idx);
    q = r_sorted.*w;
    a = y(1);
    b = y(end);

    if alpha < 1e-12
        left = y-a;
        right = b-y;
        denom = b-a;
    else
        left = sinh(alpha*(y-a));
        right = sinh(alpha*(b-y));
        denom = alpha*sinh(alpha*(b-a));
    end

    prefix_left = cumsum(left.*q);
    prefix_left_excluding_self = [0; prefix_left(1:end-1)];
    suffix_right_including_self = flipud(cumsum(flipud(right.*q)));

    u_sorted = -(right.*prefix_left_excluding_self ...
        + left.*suffix_right_including_self)/denom;
    u = u_sorted(params.green_unsort_idx);
end


function L_u_p=L_fun(u_p,tflag,params)
    kx=params.kx;
    kz=params.kz;
    omega=params.omega;
    Ny=params.Ny;
    Re=params.Re;
    Ly=params.Ly;


    u=u_p(1:Ny);
    v=u_p(Ny+1:2*Ny);
    w=u_p(1+2*Ny:3*Ny);
    p=u_p(1+3*Ny:4*Ny);

    %first and second order derative in y, make sure to have scaling factor.
    Dy=@(f) 2/Ly*chebdifft(f,1);
    Dyy=@(f) (2/Ly)^2*chebdifft(f,2);
    
    lap=@(u) Dyy(u)-(kx^2+kz^2).*u;

    U0 = params.U0;
    U1 = params.U1;
    
    if strcmp(tflag,'notransp')
        A11 =@(u) 1i*omega*u+1i*kx*U0.*u-lap(u)/Re;
        L_u_p=[A11(u)+U1.*v+1i*kx*p;
               A11(v)+Dy(p);
               A11(w)+1i*kz*p;
            -1i*kx*u-Dy(v)-1i*kz*w];

    else
        A11_adjoint=@(u)-1i*omega*u-1i*kx*U0.*u-lap(u)/Re;
        L_u_p=[A11_adjoint(u)+1i*kx*p;
               A11_adjoint(v)+U1.*u+Dy(p);
               A11_adjoint(w)+1i*kz*p;
            -1i*kx*u-Dy(v)-1i*kz*w];

    end
    %B.C. for u
    L_u_p(1)=u(1);
    L_u_p(Ny)=u(Ny);
    
    %B.C. for v
    L_u_p(Ny+1)=v(1);
    L_u_p(2*Ny)=v(Ny);

    %B.C. for w
    L_u_p(2*Ny+1)=w(1);
    L_u_p(3*Ny)=w(Ny);

end


function [H,L]=resolvent_1D(params)
    kx=params.kx;
    kz=params.kz;
    omega=params.omega;
    Ny=params.Ny;
    Re=params.Re;

    [y,DM] = chebdif(Ny,2);
    D1 = DM(1:Ny,1:Ny,1);
    D2 = DM(1:Ny,1:Ny,2);
    I = eye(Ny);
    lplc = D2 - kx.^2.*I - kz.^2.*I;

    U0 = 1-y.^2;
    U1 = -2*y;

    A11 = -1i*kx.*diag(U0) + lplc./Re;

    L = [1i*omega*I-A11,      diag(U1),  zeros(Ny),  1i*kx.*I;
        zeros(Ny),  1i*omega*I-A11,        zeros(Ny),  D1;
        zeros(Ny),  zeros(Ny),   1i*omega*I-A11,       1i*kz.*I;
        -1i*kx.*I,   -D1,        -1i*kz.*I,  zeros(Ny)];

    C = [eye(3*Ny),zeros(3*Ny,Ny)];
    [~,w]=clencurt(Ny-1);
    IWC = blkdiag(diag(w.^0.5),diag(w.^0.5),diag(w.^0.5));
    C = IWC*C;

    B = [eye(3*Ny); zeros(Ny,3*Ny)];
    IWB = blkdiag(diag(w.^(-0.5)),diag(w.^(-0.5)),diag(w.^(-0.5)));
    B = B*IWB;

    L(1,:)=[1,zeros(1,4*Ny-1)];
    L(Ny,:)=[zeros(1,Ny-1),1,zeros(1,3*Ny)];
    B(1,:)=zeros(1,3*Ny);
    B(Ny,:)=zeros(1,3*Ny);
    
    % v(0)=v(N)=0
    L(Ny+1,:)=[zeros(1,Ny),1,zeros(1,3*Ny-1)];
    L(2*Ny,:)=[zeros(1,2*Ny-1),1,zeros(1,2*Ny)];
    B(Ny+1,:)=zeros(1,3*Ny);
    B(2*Ny,:)=zeros(1,3*Ny);

    % w(0)=w(N)=0
    L(2*Ny+1,:)=[zeros(1,2*Ny),1,zeros(1,2*Ny-1)];
    L(3*Ny,:)=[zeros(1,3*Ny-1),1,zeros(1,Ny)];
    B(2*Ny+1,:)=zeros(1,3*Ny);
    B(3*Ny,:)=zeros(1,3*Ny);

    H = C*(L\B);

end
