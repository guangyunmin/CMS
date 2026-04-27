clear; clc; close all;
rng(1);

rPOD = 3;
mainModeList = [1 2 3 4];
baselineMainModes = 2;
% sensorNodes = [126];
% sensorNodes = [ 80,    240];
% sensorNodes = [ 80,  160,  240];
sensorNodes = [ 40, 80,  160,  240];
targetNode  = 126;

useMeanSubtract = false;
useTikhonov = true;
lambdaReg = 1e-10;

noiseLevels = [0 0.05 0.10 0.20];
zoomSec = 0.5;

reconstructV = true;   % true: Fy->v, false: Fz->w
saveFigures = false;


doPSDAnalysis = true;
psdDetrendMean = true;    
psdDurationSec = 10.0;    
psdPlotMaxHz = [];         
psdUseLogX = true;         


L_TOTAL   = 1.19;
BIN_COUNT = 250;
BIN_SIZE  = L_TOTAL / BIN_COUNT;

nel   = BIN_COUNT;
nn    = nel + 1;
dofpn = 4;
ndof  = nn * dofpn;
Le    = BIN_SIZE;

if isempty(targetNode)
    targetNode = floor((nn-1)/2) + 1;
end

fixedDOFs = [1:4, (ndof-3):ndof];
freeDOFs  = setdiff(1:ndof, fixedDOFs);
N_FREE    = numel(freeDOFs);


E    = 177e9;
rho  = 7990;


Iy   = 1094.685 *1e-12;
Iz   = Iy;
A    = 1.08174e-4;

RHO_FLUID = 2.274*10403;
A_DISP    = A;



dataFile = 'hcf_long_time_signals(1).txt';
if ~isfile(dataFile)
    error('未找到输入文件 "%s"，请将其放在当前工作目录下。', dataFile);
end

Fext = readmatrix(dataFile);

t = Fext(:,1);
Fy_bin = Fext(:, 2 : 1+BIN_COUNT);
Fz_bin = Fext(:, 2+BIN_COUNT : 1+2*BIN_COUNT);

Nt = length(t);
dt = t(2) - t(1);
fs = 1 / dt;

fprintf('============================================================\n');
fprintf('子结构/模型缩聚 + POD 的 4DOF 梁动力响应重构（10s FFT-PSD版）\n');
fprintf('nel = %d, nn = %d, ndof = %d\n', nel, nn, ndof);
fprintf('全模型固定后自由DOF数 N_FREE = %d\n', N_FREE);
fprintf('Nt  = %d, dt = %.6e s, fs = %.6f Hz\n', Nt, dt, fs);
fprintf('目标节点 = %d\n', targetNode);
fprintf('传感器节点 = ');
fprintf('%d ', sensorNodes);
fprintf('\nPOD阶数 rPOD = %d\n', rPOD);
fprintf('主模态敏感性工况 = ');
fprintf('%d ', mainModeList);
fprintf('\n重构方向 = %s\n', ternary(reconstructV, 'v(Fy)', 'w(Fz)'));
fprintf('PSD分析 = %s\n', ternary(doPSDAnalysis, 'ON', 'OFF'));
fprintf('PSD方法 = 前 %.2f s 单段FFT / 无窗 / x轴对数=%s\n', psdDurationSec, ternary(psdUseLogX, 'ON', 'OFF'));
fprintf('============================================================\n\n');


[Kf_full, Mf_full, Cf_full, dof2free_full] = assemble_system_4dof_with_added_mass( ...
    nel, Le, nn, ndof, E, A, Iy, Iz, rho, RHO_FLUID, A_DISP);


interfaceNodes = unique([1, sensorNodes(:).', targetNode, nn]);
interfaceNodes = sort(interfaceNodes);

subRanges = [];
for k = 1:numel(interfaceNodes)-1
    a = interfaceNodes(k);
    b = interfaceNodes(k+1);
    if b > a
        subRanges = [subRanges; a, b]; 
    end
end
nSub = size(subRanges,1);

fprintf('自动生成子结构数 nSub = %d\n', nSub);
for s = 1:nSub
    fprintf('  子结构 %2d: node %d -> node %d\n', s, subRanges(s,1), subRanges(s,2));
end
fprintf('\n');


Q_hist_full = zeros(N_FREE, Nt);

for it = 1:Nt
    Q = zeros(N_FREE,1);

    for e = 1:nel
        fy = Fy_bin(it,e) / Le;
        fz = Fz_bin(it,e) / Le;

        fe = beam4dof_force_vector_from_q(fy, fz, Le);

        dofs = [ ...
            global_dof(e,1), global_dof(e,2), global_dof(e+1,1), global_dof(e+1,2), ...
            global_dof(e,3), global_dof(e,4), global_dof(e+1,3), global_dof(e+1,4)];

        for j = 1:8
            fid = dof2free_full(dofs(j));
            if fid > 0
                Q(fid) = Q(fid) + fe(j);
            end
        end
    end

    Q_hist_full(:,it) = Q;
end

[Ufree_hist_full, ~, ~] = NewmarkBeta_MDOF(Kf_full, Mf_full, Cf_full, Q_hist_full, dt);

Ufull_hist = zeros(ndof, Nt);
Ufull_hist(freeDOFs,:) = Ufree_hist_full;

Xtrue = zeros(nn, Nt);
for node = 1:nn
    if reconstructV
        Xtrue(node,:) = Ufull_hist(global_dof(node,1), :);
    else
        Xtrue(node,:) = Ufull_hist(global_dof(node,3), :);
    end
end
y_target_true = Xtrue(targetNode,:).';


mSensors = numel(sensorNodes);
C = zeros(mSensors, nn);
for i = 1:mSensors
    C(i, sensorNodes(i)) = 1;
end
Ysensor_true = C * Xtrue;


nModeCases = numel(mainModeList);

modeResponses   = cell(nModeCases,1);
modePhiPOD      = cell(nModeCases,1);
modeXmean       = cell(nModeCases,1);
modeXsnap       = cell(nModeCases,1);
modeSingVals    = cell(nModeCases,1);
modeCumEnergy   = cell(nModeCases,1);

mode_rEff       = zeros(nModeCases,1);

mode_mse        = zeros(nModeCases,1);
mode_rmse       = zeros(nModeCases,1);
mode_nrmse      = zeros(nModeCases,1);
mode_mae        = zeros(nModeCases,1);
mode_maxae      = zeros(nModeCases,1);
mode_corr       = zeros(nModeCases,1);
mode_R2         = zeros(nModeCases,1);

mode_ndof_full  = zeros(nModeCases,1);
mode_ndof_free  = zeros(nModeCases,1);
mode_ndof_red   = zeros(nModeCases,1);
mode_red_ratio  = zeros(nModeCases,1);
mode_nBoundary  = zeros(nModeCases,1);
mode_nModal     = zeros(nModeCases,1);

mode_freq_full_hz = cell(nModeCases,1);
mode_freq_red_hz  = cell(nModeCases,1);
mode_freq_err_pct = cell(nModeCases,1);

podEnergy_base = [];
podCumEnergy_base = [];
podSingVals_base = [];


for im = 1:nModeCases
    nMainModesPerSub = mainModeList(im);

    fprintf('================ 主模态工况：%d 阶 =================\n', nMainModesPerSub);

    [Tcms, cmsInfo] = build_global_cms_basis( ...
        Kf_full, Mf_full, dof2free_full, fixedDOFs, ...
        interfaceNodes, subRanges, nMainModesPerSub);

    nRed = size(Tcms,2);

    fprintf('全模型总DOF                 = %d\n', ndof);
    fprintf('全模型固定后自由DOF         = %d\n', N_FREE);
    fprintf('缩聚后自由DOF               = %d\n', nRed);
    fprintf('  其中共享物理界面DOF数      = %d\n', cmsInfo.nBoundaryRed);
    fprintf('  其中内部主模态DOF总数      = %d\n', cmsInfo.nModalRed);
    fprintf('相对全模型自由DOF缩减率      = %.2f %%\n', 100*(1 - nRed/max(N_FREE,1)));

    Kred = Tcms' * Kf_full * Tcms;
    Mred = Tcms' * Mf_full * Tcms;
    Cred = Tcms' * Cf_full * Tcms;

    nFreqCompare = min([12, size(Kf_full,1), size(Kred,1)]);
    freq_full_hz = solve_natural_frequencies(Kf_full, Mf_full, nFreqCompare);
    freq_red_hz  = solve_natural_frequencies(Kred,    Mred,    nFreqCompare);

    nFreqCommon = min(numel(freq_full_hz), numel(freq_red_hz));
    freq_err_pct = 100 * abs(freq_red_hz(1:nFreqCommon) - freq_full_hz(1:nFreqCommon)) ...
        ./ max(abs(freq_full_hz(1:nFreqCommon)), eps);

    fprintf('固有频率对比（前 %d 阶，Hz）\n', nFreqCommon);
    fprintf('Mode | Full(Hz)    | Reduced(Hz) | RelErr(%%)\n');
    for ik = 1:nFreqCommon
        fprintf('%4d | %11.6f | %11.6f | %9.4f\n', ...
            ik, freq_full_hz(ik), freq_red_hz(ik), freq_err_pct(ik));
    end

    [loadCases, ~] = generate_static_load_cases_4dof_fixed_beam(nel, nn, Le, reconstructV);
    nCases = numel(loadCases);

    Xsnap = zeros(nn, nCases);

    for k = 1:nCases
        Fg = loadCases{k};
        if numel(Fg) ~= ndof
            error('第 %d 个静力工况长度错误：numel(Fg)=%d, ndof=%d', k, numel(Fg), ndof);
        end

        Ff = Fg(freeDOFs);

        qred = Kred \ (Tcms' * Ff);
        uf_red = Tcms * qred;

        u = zeros(ndof,1);
        u(freeDOFs) = uf_red;

        for node = 1:nn
            if reconstructV
                Xsnap(node,k) = u(global_dof(node,1));
            else
                Xsnap(node,k) = u(global_dof(node,3));
            end
        end
    end

    if useMeanSubtract
        xMean = mean(Xsnap, 2);
        X0 = Xsnap - xMean;
    else
        xMean = zeros(nn,1);
        X0 = Xsnap;
    end

    [U_pod, S_pod, ~] = svd(X0, 'econ');
    singVals = diag(S_pod);
    energy = singVals.^2 / max(sum(singVals.^2), eps);
    cumEnergy = cumsum(energy);

    rEff = min([rPOD, size(U_pod,2), numel(sensorNodes)]);
    PhiPOD = U_pod(:,1:rEff);

    Xrec = reconstruct_dynamic_field_from_pod( ...
        Ysensor_true, C, PhiPOD, xMean, useMeanSubtract, useTikhonov, lambdaReg);

    y_target_rec = Xrec(targetNode,:).';
    met = calc_metrics(y_target_true, y_target_rec);

    fprintf('MSE      = %.6e\n', met.mse);
    fprintf('RMSE     = %.6e\n', met.rmse);
    fprintf('NRMSE    = %.4f %%\n', 100*met.nrmse);
    fprintf('MAE      = %.6e\n', met.mae);
    fprintf('MaxAE    = %.6e\n', met.maxae);
    fprintf('CorrCoef = %.6f\n', met.corr);
    fprintf('R2       = %.6f\n', met.R2);
    fprintf('POD阶数   = %d\n', rEff);
    fprintf('累计能量   = %.6f\n', cumEnergy(rEff));
    fprintf('====================================================\n\n');

    modeResponses{im} = y_target_rec;
    modePhiPOD{im}    = PhiPOD;
    modeXmean{im}     = xMean;
    modeXsnap{im}     = Xsnap;
    modeSingVals{im}  = singVals;
    modeCumEnergy{im} = cumEnergy;
    mode_rEff(im)     = rEff;

    mode_mse(im)      = met.mse;
    mode_rmse(im)     = met.rmse;
    mode_nrmse(im)    = met.nrmse;
    mode_mae(im)      = met.mae;
    mode_maxae(im)    = met.maxae;
    mode_corr(im)     = met.corr;
    mode_R2(im)       = met.R2;

    mode_ndof_full(im) = ndof;
    mode_ndof_free(im) = N_FREE;
    mode_ndof_red(im)  = nRed;
    mode_red_ratio(im) = 1 - nRed/max(N_FREE,1);
    mode_nBoundary(im) = cmsInfo.nBoundaryRed;
    mode_nModal(im)    = cmsInfo.nModalRed;

    mode_freq_full_hz{im} = freq_full_hz;
    mode_freq_red_hz{im}  = freq_red_hz;
    mode_freq_err_pct{im} = freq_err_pct;
end


idxBase = find(mainModeList == baselineMainModes, 1);
if isempty(idxBase)
    idxBase = 1;
    baselineMainModes = mainModeList(1);
end

PhiPOD_base = modePhiPOD{idxBase};
xMean_base  = modeXmean{idxBase};
rEff_base   = mode_rEff(idxBase);
y_target_rec_base = modeResponses{idxBase};

podCumEnergy_base = modeCumEnergy{idxBase};
podSingVals_base  = modeSingVals{idxBase};
podEnergy_base    = podSingVals_base.^2 / max(sum(podSingVals_base.^2), eps);

fprintf('======== 基线工况：主模态 = %d 阶 ========\n', baselineMainModes);
fprintf('全模型总DOF                 = %d\n', ndof);
fprintf('全模型固定后自由DOF         = %d\n', mode_ndof_free(idxBase));
fprintf('缩聚后自由DOF               = %d\n', mode_ndof_red(idxBase));
fprintf('缩减率                      = %.2f %%\n', 100*mode_red_ratio(idxBase));
fprintf('MSE      = %.6e\n', mode_mse(idxBase));
fprintf('RMSE     = %.6e\n', mode_rmse(idxBase));
fprintf('NRMSE    = %.4f %%\n', 100*mode_nrmse(idxBase));
fprintf('MAE      = %.6e\n', mode_mae(idxBase));
fprintf('MaxAE    = %.6e\n', mode_maxae(idxBase));
fprintf('CorrCoef = %.6f\n', mode_corr(idxBase));
fprintf('R2       = %.6f\n', mode_R2(idxBase));
fprintf('POD阶数   = %d\n', rEff_base);
fprintf('=========================================\n\n');

freq_full_base = mode_freq_full_hz{idxBase};
freq_red_base  = mode_freq_red_hz{idxBase};
freq_err_base  = mode_freq_err_pct{idxBase};

nFreqBase = min([numel(freq_full_base), numel(freq_red_base), numel(freq_err_base)]);
fprintf('======== 基线工况固有频率对比（前 %d 阶） ========\n', nFreqBase);
fprintf('Mode | Full(Hz)    | Reduced(Hz) | RelErr(%%)\n');
for ik = 1:nFreqBase
    fprintf('%4d | %11.6f | %11.6f | %9.4f\n', ...
        ik, freq_full_base(ik), freq_red_base(ik), freq_err_base(ik));
end
fprintf('===============================================\n\n');

nNoise = numel(noiseLevels);
noiseResponses = cell(nNoise,1);
noise_mse   = zeros(nNoise,1);
noise_rmse  = zeros(nNoise,1);
noise_nrmse = zeros(nNoise,1);
noise_mae   = zeros(nNoise,1);
noise_maxae = zeros(nNoise,1);
noise_corr  = zeros(nNoise,1);
noise_R2    = zeros(nNoise,1);


noise10_mse   = zeros(nNoise,1);
noise10_rmse  = zeros(nNoise,1);
noise10_nrmse = zeros(nNoise,1);
noise10_mae   = zeros(nNoise,1);
noise10_maxae = zeros(nNoise,1);
noise10_corr  = zeros(nNoise,1);
noise10_R2    = zeros(nNoise,1);

noisePSD = cell(nNoise,1);
f_psd_ref = [];
S_true_ref = [];
S_base_ref = [];

idxPSD = find(t <= (t(1) + psdDurationSec));
if numel(idxPSD) < 2
    error('前 %.2f s 内采样点不足，无法进行 10 s 窗口统计。', psdDurationSec);
end
t_psd = t(idxPSD);

for i = 1:nNoise
    nl = noiseLevels(i);

    Ysensor_noisy = zeros(size(Ysensor_true));
    for s = 1:mSensors
        Ysensor_noisy(s,:) = add_rms_noise(Ysensor_true(s,:).', nl).';
    end

    Xrec_i = reconstruct_dynamic_field_from_pod( ...
        Ysensor_noisy, C, PhiPOD_base, xMean_base, useMeanSubtract, useTikhonov, lambdaReg);

    yrec_i = Xrec_i(targetNode,:).';
    noiseResponses{i} = yrec_i;

    met = calc_metrics(y_target_true, yrec_i);
    noise_mse(i)   = met.mse;
    noise_rmse(i)  = met.rmse;
    noise_nrmse(i) = met.nrmse;
    noise_mae(i)   = met.mae;
    noise_maxae(i) = met.maxae;
    noise_corr(i)  = met.corr;
    noise_R2(i)    = met.R2;

 
    met10 = calc_metrics(y_target_true(idxPSD), yrec_i(idxPSD));
    noise10_mse(i)   = met10.mse;
    noise10_rmse(i)  = met10.rmse;
    noise10_nrmse(i) = met10.nrmse;
    noise10_mae(i)   = met10.mae;
    noise10_maxae(i) = met10.maxae;
    noise10_corr(i)  = met10.corr;
    noise10_R2(i)    = met10.R2;

    if doPSDAnalysis
        y_true_psd = y_target_true(idxPSD);
        y_rec_psd  = yrec_i(idxPSD);
        y_err_psd  = y_true_psd - y_rec_psd;

        [f_psd, S_true] = simple_fft_psd(y_true_psd, fs, psdDetrendMean);
        [~,     S_rec ] = simple_fft_psd(y_rec_psd,  fs, psdDetrendMean);
        [~,     S_err ] = simple_fft_psd(y_err_psd,  fs, psdDetrendMean);

        peak_true = extract_psd_peak_positive(f_psd, S_true);
        peak_rec  = extract_psd_peak_positive(f_psd, S_rec);
        peak_err  = extract_psd_peak_positive(f_psd, S_err);

        noisePSD{i}.noiseLevel    = nl;
        noisePSD{i}.t_psd         = t_psd;
        noisePSD{i}.f             = f_psd;
        noisePSD{i}.S_true        = S_true;
        noisePSD{i}.S_rec         = S_rec;
        noisePSD{i}.S_err         = S_err;
        noisePSD{i}.peak_true_hz  = peak_true.freqHz;
        noisePSD{i}.peak_rec_hz   = peak_rec.freqHz;
        noisePSD{i}.peak_err_hz   = peak_err.freqHz;
        noisePSD{i}.peak_true_psd = peak_true.amp;
        noisePSD{i}.peak_rec_psd  = peak_rec.amp;
        noisePSD{i}.peak_err_psd  = peak_err.amp;
        noisePSD{i}.psd_l2_relerr = norm(S_rec - S_true) / max(norm(S_true), eps);

        if isempty(f_psd_ref)
            f_psd_ref = f_psd;
            S_true_ref = S_true;
        end
        if abs(nl) < eps
            S_base_ref = S_rec;
        end
    end
end


psdPlotData = struct([]);
if doPSDAnalysis
    if isempty(psdPlotMaxHz)
        psdPlotMaxHz = 0.4 * (fs/2);
    end

    for i = 1:nNoise
        fplt = noisePSD{i}.f;
        Strue = noisePSD{i}.S_true;
        Srec  = noisePSD{i}.S_rec;
        Serr  = noisePSD{i}.S_err;

        idxHz = (fplt > 0) & (fplt <= psdPlotMaxHz);
        if ~any(idxHz)
            idxHz = (fplt > 0);
        end

        psdPlotData(i).noiseLevel   = noiseLevels(i);
        psdPlotData(i).f_plot       = fplt(idxHz);
        psdPlotData(i).Strue_plot   = Strue(idxHz);
        psdPlotData(i).Srec_plot    = Srec(idxHz);
        psdPlotData(i).Serr_plot    = Serr(idxHz);
        psdPlotData(i).Strue_dB     = 10*log10(Strue(idxHz) + eps);
        psdPlotData(i).Srec_dB      = 10*log10(Srec(idxHz)  + eps);
        psdPlotData(i).Serr_dB      = 10*log10(Serr(idxHz)  + eps);
        psdPlotData(i).xScale       = ternary(psdUseLogX, 'log', 'linear');
        psdPlotData(i).durationSec  = psdDurationSec;
        psdPlotData(i).plotMaxHz    = psdPlotMaxHz;
    end
else
    psdPlotData = [];
end


figure('Name','基线工况目标点动态响应重构对比','Color','w');
subplot(2,1,1);
plot(t, y_target_true, 'r-', 'LineWidth', 0.9); hold on;
plot(t, y_target_rec_base, 'b--', 'LineWidth', 1.0);
grid on;
xlim([t(1) t(end)]);
xlabel('t / s');
ylabel('位移 / m');
title(sprintf('全时程目标点动态位移对比（主模态=%d）', baselineMainModes));
legend('理论值', 'POD重构值', 'Location', 'best');

subplot(2,1,2);
plot(t, y_target_true, 'r-', 'LineWidth', 0.9); hold on;
plot(t, y_target_rec_base, 'b--', 'LineWidth', 1.0);
grid on;
xlim([t(1) t(1)+zoomSec]);
xlabel('t / s');
ylabel('位移 / m');
title(sprintf('前 %.2f s 局部放大（主模态=%d）', zoomSec, baselineMainModes));
legend('理论值', 'POD重构值', 'Location', 'best');


figure('Name','不同噪声等级下目标点重构','Color','w');
for i = 1:nNoise
    subplot(2,2,i);
    plot(t, y_target_true, 'r-', 'LineWidth', 0.9); hold on;
    plot(t, noiseResponses{i}, 'b--', 'LineWidth', 1.0);
    grid on;
    xlim([t(1) t(1)+zoomSec]);
    xlabel('t / s');
    ylabel('位移 / m');
    title(sprintf('Noise = %d%%', round(100*noiseLevels(i))));
    legend('理论值', 'POD重构值', 'Location', 'best');
end

figure('Name','噪声等级对应误差指标','Color','w');
subplot(1,4,1);
plot(100*noiseLevels, noise_mse, 'o-', 'LineWidth', 1.2);
grid on; xlabel('噪声等级 RMS / %'); ylabel('MSE'); title('MSE');

subplot(1,4,2);
plot(100*noiseLevels, noise_mae, 's-', 'LineWidth', 1.2);
grid on; xlabel('噪声等级 RMS / %'); ylabel('MAE'); title('MAE');

subplot(1,4,3);
plot(100*noiseLevels, noise_corr, 'd-', 'LineWidth', 1.2);
grid on; xlabel('噪声等级 RMS / %'); ylabel('Corr'); title('相关系数');

subplot(1,4,4);
plot(100*noiseLevels, noise_R2, '^-', 'LineWidth', 1.2);
grid on; xlabel('噪声等级 RMS / %'); ylabel('R^2'); title('R^2');


if doPSDAnalysis
    if isempty(psdPlotMaxHz)
        psdPlotMaxHz = 0.4 * (fs/2);
    end

    figure('Name','不同噪声等级下目标点响应PSD对比（前10s）','Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    for i = 1:nNoise
        nexttile;
        fplt = noisePSD{i}.f;
        Strue = noisePSD{i}.S_true;
        Srec  = noisePSD{i}.S_rec;

        idxHz = (fplt > 0) & (fplt <= psdPlotMaxHz);

        plot(fplt(idxHz), 10*log10(Strue(idxHz) + eps), 'r-', 'LineWidth', 1.0); hold on;
        plot(fplt(idxHz), 10*log10(Srec(idxHz)  + eps), 'b--', 'LineWidth', 1.0);
        grid on;
        xlabel('频率 / Hz');
        ylabel('PSD / dB');
        title(sprintf('Noise = %d%%（前 %.1f s）', round(100*noiseLevels(i)), psdDurationSec));
        legend('理论值', '重构值', 'Location', 'best');
        if psdUseLogX
            set(gca, 'XScale', 'log');
        end
    end

    figure('Name','不同噪声等级下重构误差PSD（前10s）','Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    for i = 1:nNoise
        nexttile;
        fplt = noisePSD{i}.f;
        Serr = noisePSD{i}.S_err;
        idxHz = (fplt > 0) & (fplt <= psdPlotMaxHz);

        plot(fplt(idxHz), 10*log10(Serr(idxHz) + eps), 'k-', 'LineWidth', 1.0);
        grid on;
        xlabel('频率 / Hz');
        ylabel('Error PSD / dB');
        title(sprintf('Noise = %d%%（前 %.1f s）', round(100*noiseLevels(i)), psdDurationSec));
        if psdUseLogX
            set(gca, 'XScale', 'log');
        end
    end

    psd_l2_relerr = zeros(nNoise,1);
    psd_peak_true_hz = zeros(nNoise,1);
    psd_peak_rec_hz  = zeros(nNoise,1);
    psd_peak_err_hz  = zeros(nNoise,1);
    for i = 1:nNoise
        psd_l2_relerr(i) = noisePSD{i}.psd_l2_relerr;
        psd_peak_true_hz(i) = noisePSD{i}.peak_true_hz;
        psd_peak_rec_hz(i)  = noisePSD{i}.peak_rec_hz;
        psd_peak_err_hz(i)  = noisePSD{i}.peak_err_hz;
    end

    figure('Name','PSD指标随噪声变化（前10s）','Color','w');
    subplot(1,3,1);
    plot(100*noiseLevels, psd_l2_relerr, 'o-', 'LineWidth', 1.2);
    grid on; xlabel('噪声等级 RMS / %'); ylabel('相对PSD误差');
    title('||S_{rec}-S_{true}||_2 / ||S_{true}||_2');

    subplot(1,3,2);
    plot(100*noiseLevels, psd_peak_true_hz, 'r-o', 'LineWidth', 1.2); hold on;
    plot(100*noiseLevels, psd_peak_rec_hz,  'b-s', 'LineWidth', 1.2);
    grid on; xlabel('噪声等级 RMS / %'); ylabel('主峰频率 / Hz');
    title('理论/重构 PSD 主峰频率');
    legend('理论值','重构值','Location','best');

    subplot(1,3,3);
    plot(100*noiseLevels, psd_peak_err_hz, 'k-d', 'LineWidth', 1.2);
    grid on; xlabel('噪声等级 RMS / %'); ylabel('误差主峰频率 / Hz');
    title('误差 PSD 主峰频率');
end


figure('Name','前10s误差指标随噪声变化','Color','w');
subplot(1,4,1);
plot(100*noiseLevels, noise10_mse, 'o-', 'LineWidth', 1.2);
grid on; xlabel('噪声等级 RMS / %'); ylabel('MSE_{10s}'); title('前10s MSE');

subplot(1,4,2);
plot(100*noiseLevels, noise10_mae, 's-', 'LineWidth', 1.2);
grid on; xlabel('噪声等级 RMS / %'); ylabel('MAE_{10s}'); title('前10s MAE');

subplot(1,4,3);
plot(100*noiseLevels, noise10_corr, 'd-', 'LineWidth', 1.2);
grid on; xlabel('噪声等级 RMS / %'); ylabel('Corr_{10s}'); title('前10s Corr');

subplot(1,4,4);
plot(100*noiseLevels, noise10_R2, '^-', 'LineWidth', 1.2);
grid on; xlabel('噪声等级 RMS / %'); ylabel('R^2_{10s}'); title('前10s R^2');


figure('Name','不同主模态数量下目标点响应重构','Color','w');
for i = 1:nModeCases
    subplot(2,2,i);
    plot(t, y_target_true, 'r-', 'LineWidth', 0.9); hold on;
    plot(t, modeResponses{i}, 'b--', 'LineWidth', 1.0);
    grid on;
    xlim([t(1) t(1)+zoomSec]);
    xlabel('t / s');
    ylabel('位移 / m');
    title(sprintf('主模态数 = %d', mainModeList(i)));
    legend('理论值', 'POD重构值', 'Location', 'best');
end

figure('Name','主模态敏感性指标','Color','w');
subplot(1,5,1);
plot(mainModeList, mode_mse, 'o-', 'LineWidth', 1.2);
grid on; xlabel('主模态数量'); ylabel('MSE'); title('MSE');

subplot(1,5,2);
plot(mainModeList, mode_mae, 's-', 'LineWidth', 1.2);
grid on; xlabel('主模态数量'); ylabel('MAE'); title('MAE');

subplot(1,5,3);
plot(mainModeList, mode_corr, 'd-', 'LineWidth', 1.2);
grid on; xlabel('主模态数量'); ylabel('Corr'); title('Corr');

subplot(1,5,4);
plot(mainModeList, mode_R2, '^-', 'LineWidth', 1.2);
grid on; xlabel('主模态数量'); ylabel('R^2'); title('R^2');

subplot(1,5,5);
plot(mainModeList, mode_ndof_red, 'p-', 'LineWidth', 1.2);
grid on; xlabel('主模态数量'); ylabel('缩聚后DOF'); title('缩聚后自由度');


figure('Name','基线工况缩减前后固有频率对比','Color','w');

subplot(1,2,1);
nfb = min(numel(freq_full_base), numel(freq_red_base));
plot(1:nfb, freq_full_base(1:nfb), 'o-', 'LineWidth', 1.2); hold on;
plot(1:nfb, freq_red_base(1:nfb),  's--', 'LineWidth', 1.2);
grid on;
xlabel('模态阶数');
ylabel('固有频率 / Hz');
title(sprintf('主模态=%d 时前 %d 阶频率对比', baselineMainModes, nfb));
legend('缩减前', '缩减后', 'Location', 'best');

subplot(1,2,2);
plot(1:numel(freq_err_base), freq_err_base, 'd-', 'LineWidth', 1.2);
grid on;
xlabel('模态阶数');
ylabel('相对误差 / %');
title('缩减前后固有频率相对误差');


figure('Name','基线工况POD能量占比','Color','w');

nPodBase = numel(podEnergy_base);

subplot(1,2,1);
bar(1:nPodBase, 100*podEnergy_base, 0.6);
grid on;
xlabel('POD模态阶数');
ylabel('单阶能量占比 / %');
title(sprintf('基线工况 POD 单阶能量占比（主模态=%d）', baselineMainModes));

subplot(1,2,2);
plot(1:nPodBase, 100*podCumEnergy_base, 'o-', 'LineWidth', 1.2);
grid on;
xlabel('POD模态阶数');
ylabel('累计能量占比 / %');
title(sprintf('基线工况 POD 累计能量占比（r_{POD}=%d）', rEff_base));
ylim([0 100]);


fprintf('================ 主模态敏感性汇总 ================\n');
fprintf('Modes | FullDOF | FreeDOF | RedDOF | MSE         | MAE         | Corr     | R2\n');
for i = 1:nModeCases
    fprintf('%5d | %7d | %7d | %6d | %.6e | %.6e | %.6f | %.6f\n', ...
        mainModeList(i), mode_ndof_full(i), mode_ndof_free(i), mode_ndof_red(i), ...
        mode_mse(i), mode_mae(i), mode_corr(i), mode_R2(i));
end
fprintf('==================================================\n\n');

fprintf('================ 固有频率对比汇总（基线） ================\n');
fprintf('Mode | Full(Hz)    | Reduced(Hz) | RelErr(%%)\n');
for ik = 1:nFreqBase
    fprintf('%4d | %11.6f | %11.6f | %9.4f\n', ...
        ik, freq_full_base(ik), freq_red_base(ik), freq_err_base(ik));
end
fprintf('========================================================\n\n');

fprintf('================ 前 %.2f s 时域指标汇总 ================\n', psdDurationSec);
fprintf('Noise(%%) | MSE_10s     | MAE_10s     | Corr_10s  | R2_10s\n');
for i = 1:nNoise
    fprintf('%8.2f | %.6e | %.6e | %.6f | %.6f\n', ...
        100*noiseLevels(i), noise10_mse(i), noise10_mae(i), noise10_corr(i), noise10_R2(i));
end
fprintf('=======================================================\n\n');

if doPSDAnalysis
    fprintf('================ 前 %.2f s PSD 汇总 ================\n', psdDurationSec);
    fprintf('Noise(%%) | PeakTrue(Hz) | PeakRec(Hz) | PeakErr(Hz) | RelPSDerr\n');
    for i = 1:nNoise
        fprintf('%8.2f | %12.6f | %11.6f | %11.6f | %.6e\n', ...
            100*noiseLevels(i), ...
            noisePSD{i}.peak_true_hz, noisePSD{i}.peak_rec_hz, ...
            noisePSD{i}.peak_err_hz, noisePSD{i}.psd_l2_relerr);
    end
    fprintf('====================================================\n\n');
end


resultSubPOD.t                 = t;
resultSubPOD.fs                = fs;
resultSubPOD.Xtrue             = Xtrue;
resultSubPOD.y_target_true     = y_target_true;
resultSubPOD.sensorNodes       = sensorNodes;
resultSubPOD.targetNode        = targetNode;
resultSubPOD.interfaceNodes    = interfaceNodes;
resultSubPOD.subRanges         = subRanges;
resultSubPOD.mainModeList      = mainModeList;

resultSubPOD.modeResponses     = modeResponses;
resultSubPOD.modePhiPOD        = modePhiPOD;
resultSubPOD.modeXmean         = modeXmean;
resultSubPOD.modeXsnap         = modeXsnap;
resultSubPOD.modeSingVals      = modeSingVals;
resultSubPOD.modeCumEnergy     = modeCumEnergy;
resultSubPOD.mode_rEff         = mode_rEff;

resultSubPOD.mode_mse          = mode_mse;
resultSubPOD.mode_rmse         = mode_rmse;
resultSubPOD.mode_nrmse        = mode_nrmse;
resultSubPOD.mode_mae          = mode_mae;
resultSubPOD.mode_maxae        = mode_maxae;
resultSubPOD.mode_corr         = mode_corr;
resultSubPOD.mode_R2           = mode_R2;

resultSubPOD.mode_ndof_full    = mode_ndof_full;
resultSubPOD.mode_ndof_free    = mode_ndof_free;
resultSubPOD.mode_ndof_red     = mode_ndof_red;
resultSubPOD.mode_red_ratio    = mode_red_ratio;
resultSubPOD.mode_nBoundary    = mode_nBoundary;
resultSubPOD.mode_nModal       = mode_nModal;

resultSubPOD.mode_freq_full_hz = mode_freq_full_hz;
resultSubPOD.mode_freq_red_hz  = mode_freq_red_hz;
resultSubPOD.mode_freq_err_pct = mode_freq_err_pct;

resultSubPOD.baselineMainModes = baselineMainModes;
resultSubPOD.y_target_rec_base = y_target_rec_base;

resultSubPOD.noiseLevels       = noiseLevels;
resultSubPOD.noiseResponses    = noiseResponses;
resultSubPOD.noise_mse         = noise_mse;
resultSubPOD.noise_rmse        = noise_rmse;
resultSubPOD.noise_nrmse       = noise_nrmse;
resultSubPOD.noise_mae         = noise_mae;
resultSubPOD.noise_maxae       = noise_maxae;
resultSubPOD.noise_corr        = noise_corr;
resultSubPOD.noise_R2          = noise_R2;

resultSubPOD.noise10_mse       = noise10_mse;
resultSubPOD.noise10_rmse      = noise10_rmse;
resultSubPOD.noise10_nrmse     = noise10_nrmse;
resultSubPOD.noise10_mae       = noise10_mae;
resultSubPOD.noise10_maxae     = noise10_maxae;
resultSubPOD.noise10_corr      = noise10_corr;
resultSubPOD.noise10_R2        = noise10_R2;

resultSubPOD.ndof_full         = ndof;
resultSubPOD.ndof_free         = N_FREE;
resultSubPOD.ndof_reduced_base = mode_ndof_red(idxBase);

resultSubPOD.freq_full_base_hz = freq_full_base;
resultSubPOD.freq_red_base_hz  = freq_red_base;
resultSubPOD.freq_err_base_pct = freq_err_base;

resultSubPOD.podEnergy_base    = podEnergy_base;
resultSubPOD.podCumEnergy_base = podCumEnergy_base;
resultSubPOD.podSingVals_base  = podSingVals_base;

resultSubPOD.doPSDAnalysis     = doPSDAnalysis;
resultSubPOD.psdDetrendMean    = psdDetrendMean;
resultSubPOD.psdDurationSec    = psdDurationSec;
resultSubPOD.psdUseLogX        = psdUseLogX;
resultSubPOD.noisePSD          = noisePSD;
resultSubPOD.psdPlotData       = psdPlotData;
resultSubPOD.f_psd_ref         = f_psd_ref;
resultSubPOD.S_true_ref        = S_true_ref;
resultSubPOD.S_base_ref        = S_base_ref;

save('paper_substructure_pod_fulltruth_sensitivity_v8_10s_logx_psd_with_R2_result.mat', 'resultSubPOD');
fprintf('已保存: paper_substructure_pod_fulltruth_sensitivity_v8_10s_logx_psd_with_R2_result.mat\n');

if saveFigures
    figs = findall(groot,'Type','figure');
    for k = 1:numel(figs)
        figNum = figs(k).Number;
        saveas(figs(k), sprintf('figure_%02d.png', figNum));
    end
end

%% ========================= 局部函数区 =========================

function [Tcms, info] = build_global_cms_basis( ...
    Kf_full, Mf_full, dof2free_full, fixedDOFs, ...
    interfaceNodes, subRanges, nMainModesPerSub)

N_FREE = size(Kf_full,1);

boundaryFreeIdxList = [];
for n = interfaceNodes(2:end-1)
    for ld = 1:4
        gd = global_dof(n, ld);
        if ~ismember(gd, fixedDOFs)
            fid = dof2free_full(gd);
            if fid > 0
                boundaryFreeIdxList(end+1,1) = fid; 
            end
        end
    end
end
boundaryFreeIdxList = unique(boundaryFreeIdxList, 'stable');
nBoundaryRed = numel(boundaryFreeIdxList);

boundaryColMap = containers.Map('KeyType','double','ValueType','double');
for i = 1:nBoundaryRed
    boundaryColMap(boundaryFreeIdxList(i)) = i;
end

T_boundary = zeros(N_FREE, nBoundaryRed);
for i = 1:nBoundaryRed
    T_boundary(boundaryFreeIdxList(i), i) = 1.0;
end

T_modal = [];

for s = 1:size(subRanges,1)
    nodeA = subRanges(s,1);
    nodeB = subRanges(s,2);
    nodes = nodeA:nodeB;

    allGlobalDofs = [];
    for n = nodes
        for ld = 1:4
            gd = global_dof(n, ld);
            if ~ismember(gd, fixedDOFs)
                allGlobalDofs(end+1,1) = gd; 
            end
        end
    end

    boundaryGlobalDofs = [];
    for n = [nodeA, nodeB]
        for ld = 1:4
            gd = global_dof(n, ld);
            if ~ismember(gd, fixedDOFs)
                boundaryGlobalDofs(end+1,1) = gd; 
            end
        end
    end
    boundaryGlobalDofs = unique(boundaryGlobalDofs, 'stable');

    internalGlobalDofs = setdiff(allGlobalDofs, boundaryGlobalDofs, 'stable');

    bFree = dof2free_full(boundaryGlobalDofs);
    iFree = dof2free_full(internalGlobalDofs);
    localFree = [bFree; iFree];

    nb = numel(bFree);
    ni = numel(iFree);

    if ni == 0
        continue;
    end

    Ksub = Kf_full(localFree, localFree);
    Msub = Mf_full(localFree, localFree);

    Kib = Ksub(nb+1:end, 1:nb);
    Kii = Ksub(nb+1:end, nb+1:end);
    Mii = Msub(nb+1:end, nb+1:end);

    Psi = -Kii \ Kib;

    [V,D] = eig((Kii+Kii')/2, (Mii+Mii')/2);
    lam = real(diag(D));
    valid = isfinite(lam) & lam > 0;
    lam = lam(valid);
    V   = V(:,valid);

    [lam, idx] = sort(lam, 'ascend');
    V = V(:,idx);

    nKeep = min([nMainModesPerSub, size(V,2)]);
    Phi = V(:,1:nKeep);

    for k = 1:nKeep
        mk = Phi(:,k)' * Mii * Phi(:,k);
        if mk > eps
            Phi(:,k) = Phi(:,k) / sqrt(mk);
        end
    end

    for jb = 1:nb
        globalBoundaryFid = bFree(jb);
        if isKey(boundaryColMap, globalBoundaryFid)
            col = boundaryColMap(globalBoundaryFid);
            T_boundary(iFree, col) = T_boundary(iFree, col) + Psi(:,jb);
        end
    end

    Tm_sub = zeros(N_FREE, nKeep);
    Tm_sub(iFree, :) = Phi;
    T_modal = [T_modal, Tm_sub]; 
end

Tcms = [T_boundary, T_modal];

info.nBoundaryRed = nBoundaryRed;
info.nModalRed    = size(T_modal,2);
end

function [Kf, Mf, Cf, dof2free] = assemble_system_4dof_with_added_mass( ...
    nel, Le, nn, ndof, E, A, Iy, Iz, rho, rhoFluid, ADisp)

N_FREE = ndof - 8;

dof2free = -ones(ndof,1);
k = 1;
for n = 1:nn
    for loc = 0:3
        glob = (n-1)*4 + loc + 1;
        if n==1 || n==nn
            dof2free(glob) = -1;
        else
            dof2free(glob) = k;
            k = k + 1;
        end
    end
end

Kf        = zeros(N_FREE, N_FREE);
Mf_struct = zeros(N_FREE, N_FREE);
Mf_add    = zeros(N_FREE, N_FREE);

for e = 1:nel
    [Ke, Me] = beam4dof_element_matrices(Le, E, A, Iy, Iz, rho, 0, 0);
    [~, Me_all] = beam4dof_element_matrices(Le, E, A, Iy, Iz, rho, rhoFluid, ADisp);
    Me_add = Me_all - Me;

    dofs = [ ...
        global_dof(e,1), global_dof(e,2), global_dof(e+1,1), global_dof(e+1,2), ...
        global_dof(e,3), global_dof(e,4), global_dof(e+1,3), global_dof(e+1,4)];

    for i = 1:8
        fi = dof2free(dofs(i));
        if fi < 1, continue; end
        for j = 1:8
            fj = dof2free(dofs(j));
            if fj < 1, continue; end
            Kf(fi,fj)        = Kf(fi,fj)        + Ke(i,j);
            Mf_struct(fi,fj) = Mf_struct(fi,fj) + Me(i,j);
            Mf_add(fi,fj)    = Mf_add(fi,fj)    + Me_add(i,j);
        end
    end
end

Mf = Mf_struct + Mf_add;
Cf = build_rayleigh_damping(Kf, Mf, 0.013);
end

function [Ke, Me] = beam4dof_element_matrices(Le, E, A, Iy, Iz, rho, rhoFluid, ADisp)
kv = [12       6*Le   -12       6*Le;
      6*Le   4*Le^2   -6*Le   2*Le^2;
     -12      -6*Le    12      -6*Le;
      6*Le   2*Le^2   -6*Le   4*Le^2];

kw = kv;

vidx = [1 2 3 4];
widx = [5 6 7 8];

Ke = zeros(8,8);
Ke(vidx,vidx) = kv * E*Iz / Le^3;
Ke(widx,widx) = kw * E*Iy / Le^3;

M4 = [156       22*Le    54      -13*Le;
      22*Le   4*Le^2   13*Le    -3*Le^2;
      54        13*Le   156      -22*Le;
     -13*Le   -3*Le^2  -22*Le    4*Le^2];

Me = zeros(8,8);
Me(vidx,vidx) = M4 * rho*A*Le/420;
Me(widx,widx) = M4 * rho*A*Le/420;

Me_add = zeros(8,8);
Me_add(vidx,vidx) = M4 * rhoFluid*ADisp*Le/420;
Me_add(widx,widx) = M4 * rhoFluid*ADisp*Le/420;

Me = Me + Me_add;
end

function C = build_rayleigh_damping(K, M, zeta)
[V,D] = eig((K+K')/2, (M+M')/2);
lam = real(diag(D));
lam = sort(lam(isfinite(lam) & lam > 0), 'ascend');

if isempty(lam)
    C = zeros(size(K));
    return;
end

w1 = sqrt(lam(1));
w2 = sqrt(lam(min(3,numel(lam))));

a0 = 2*zeta*w1*w2/(w1+w2);
a1 = 2*zeta/(w1+w2);

C = a0*M + a1*K;
end

function gdof = global_dof(node, localDof)
gdof = (node-1)*4 + localDof;
end

function fe = beam4dof_force_vector_from_q(fy, fz, L)
fe_v = fy * [L/2; L^2/12; L/2; -L^2/12];
fe_w = fz * [L/2; L^2/12; L/2; -L^2/12];
fe = [fe_v; fe_w];
end

function [loadCases, caseInfo] = generate_static_load_cases_4dof_fixed_beam(nel, nn, Le, useV)
loadCases = {};
caseInfo  = {};

dirDOF = 1;
dirChar = 'v';
if ~useV
    dirDOF = 3;
    dirChar = 'w';
end

pointNodes = unique(round([0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]*(nn-1))+1);
pointNodes = pointNodes(pointNodes > 1 & pointNodes < nn);

pointLoads = linspace(-5e3, -5e4, 6);
for i = 1:numel(pointNodes)
    for j = 1:numel(pointLoads)
        F = zeros(nn*4,1);
        F(global_dof(pointNodes(i), dirDOF)) = pointLoads(j);
        loadCases{end+1} = F; 
        caseInfo{end+1} = sprintf('Point load at node %d, %.2e N in %s', ...
            pointNodes(i), pointLoads(j), dirChar); 
    end
end

qVals = linspace(-1e3, -8e3, 6);
for i = 1:numel(qVals)
    F = assemble_uniform_distributed_load_segment_4dof(1, nel, nel, nn, Le, qVals(i), dirChar);
    loadCases{end+1} = F; 
    caseInfo{end+1} = sprintf('Full-span UDL %.2e N/m in %s', qVals(i), dirChar); 
end

for i = 1:numel(qVals)
    F1 = assemble_uniform_distributed_load_segment_4dof(1, floor(nel/2), nel, nn, Le, qVals(i), dirChar);
    loadCases{end+1} = F1; 
    caseInfo{end+1} = sprintf('Left-half UDL %.2e N/m in %s', qVals(i), dirChar); %#ok<AGROW>

    F2 = assemble_uniform_distributed_load_segment_4dof(floor(nel/2)+1, nel, nel, nn, Le, qVals(i), dirChar);
    loadCases{end+1} = F2; 
    caseInfo{end+1} = sprintf('Right-half UDL %.2e N/m in %s', qVals(i), dirChar); %#ok<AGROW>
end
end

function F = assemble_uniform_distributed_load_segment_4dof(eStart, eEnd, nelTotal, nn, Le, q, dirFlag)
F = zeros(nn*4,1);

if eStart > eEnd
    error('eStart 不能大于 eEnd');
end
if eStart < 1 || eEnd > nelTotal
    error('均布载荷区间超出单元范围');
end

for e = eStart:eEnd
    switch lower(dirFlag)
        case 'v'
            fe = [q*Le/2; q*Le^2/12; q*Le/2; -q*Le^2/12; 0; 0; 0; 0];
        case 'w'
            fe = [0; 0; 0; 0; q*Le/2; q*Le^2/12; q*Le/2; -q*Le^2/12];
        otherwise
            error('dirFlag 只能是 v 或 w');
    end

    dofs = [ ...
        global_dof(e,1), global_dof(e,2), global_dof(e+1,1), global_dof(e+1,2), ...
        global_dof(e,3), global_dof(e,4), global_dof(e+1,3), global_dof(e+1,4)];

    F(dofs) = F(dofs) + fe;
end
end

function [uu, vv, aa] = NewmarkBeta_MDOF(K, M, C, Q_hist, dt)
beta  = 0.25;
gamma = 0.5;

nstep = size(Q_hist,2);
NDOF  = size(M,1);

uu = zeros(NDOF,nstep);
vv = zeros(NDOF,nstep);
aa = zeros(NDOF,nstep);

aa(:,1) = M \ (Q_hist(:,1) - C*vv(:,1) - K*uu(:,1));

a0 = 1/(beta*dt^2);
a1 = gamma/(beta*dt);
a2 = 1/(beta*dt);
a3 = 1/(2*beta)-1;
a4 = gamma/beta-1;
a5 = dt*(gamma/(2*beta)-1);
a6 = dt*(1-gamma);
a7 = dt*gamma;

Keff = K + a0*M + a1*C;

for k = 1:nstep-1
    Qeff = Q_hist(:,k+1) + ...
           M*(a0*uu(:,k)+a2*vv(:,k)+a3*aa(:,k)) + ...
           C*(a1*uu(:,k)+a4*vv(:,k)+a5*aa(:,k));

    uu(:,k+1) = Keff \ Qeff;
    aa(:,k+1) = a0*(uu(:,k+1)-uu(:,k)) - a2*vv(:,k) - a3*aa(:,k);
    vv(:,k+1) = vv(:,k) + a6*aa(:,k) + a7*aa(:,k+1);
end
end

function Xrec = reconstruct_dynamic_field_from_pod( ...
    Ysensor, C, PhiPOD, xMean, useMeanSubtract, useTikhonov, lambdaReg)

Nt = size(Ysensor,2);
nState = size(PhiPOD,1);
rEff = size(PhiPOD,2);

Xrec = zeros(nState, Nt);
H = C * PhiPOD;

for it = 1:Nt
    y = Ysensor(:,it);

    if useMeanSubtract
        rhs = y - C*xMean;
    else
        rhs = y;
    end

    if useTikhonov
        a = (H' * H + lambdaReg * eye(rEff)) \ (H' * rhs);
    else
        a = pinv(H) * rhs;
    end

    if useMeanSubtract
        xrec = xMean + PhiPOD * a;
    else
        xrec = PhiPOD * a;
    end

    Xrec(:,it) = xrec;
end
end

function yNoisy = add_rms_noise(y, noiseLevel)
y = y(:);
ny = randn(size(y));
s = std(ny);
if s < eps
    yNoisy = y;
    return;
end
ny = ny / s;
ny = noiseLevel * rms(y) * ny;
yNoisy = y + ny;
end

function met = calc_metrics(yTrue, yPred)
yTrue = yTrue(:);
yPred = yPred(:);
e  = yTrue - yPred;
ae = abs(e);

met.mse   = mean(e.^2);
met.rmse  = sqrt(met.mse);
met.nrmse = met.rmse / max(rms(yTrue), eps);
met.mae   = mean(ae);
met.maxae = max(ae);

C = corrcoef(yTrue, yPred);
if numel(C) >= 4
    met.corr = C(1,2);
else
    met.corr = NaN;
end

SS_res = sum((yTrue - yPred).^2);
SS_tot = sum((yTrue - mean(yTrue)).^2);
met.R2 = 1 - SS_res / max(SS_tot, eps);
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

function freq_hz = solve_natural_frequencies(K, M, nModes)
K = (K + K') / 2;
M = (M + M') / 2;

try
    opts.disp = 0;
    [~, D] = eigs(K, M, nModes, 'smallestabs', opts);
    lam = real(diag(D));
catch
    [~, D] = eig(K, M);
    lam = real(diag(D));
end

lam = lam(isfinite(lam) & lam > 0);
lam = sort(lam, 'ascend');

nKeep = min(nModes, numel(lam));
freq_hz = sqrt(lam(1:nKeep)) / (2*pi);
end


function [f, Pxx] = simple_fft_psd(x, fs, doDetrendMean)
x = x(:);
N = length(x);

if N < 2
    error('信号长度过短，无法进行 PSD 计算。');
end

if doDetrendMean
    x = x - mean(x);
end

X = fft(x);
P2 = abs(X).^2 / (fs * N);   % 双边功率谱密度

if rem(N,2) == 0
    nPos = N/2 + 1;
    Pxx = P2(1:nPos);
    if nPos > 2
        Pxx(2:end-1) = 2 * Pxx(2:end-1);
    end
else
    nPos = (N+1)/2;
    Pxx = P2(1:nPos);
    if nPos > 1
        Pxx(2:end) = 2 * Pxx(2:end);
    end
end

f = (0:nPos-1).' * fs / N;
end

function peak = extract_psd_peak_positive(f, Pxx)
idx = find(f > 0);
if isempty(idx)
    peak.amp = NaN;
    peak.freqHz = NaN;
    return;
end

Pp = Pxx(idx);
fp = f(idx);
[amp, k] = max(Pp);

peak.amp = amp;
peak.freqHz = fp(k);
end
