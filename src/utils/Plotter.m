classdef Plotter < handle
%PLOTTER Class to display multidimensional images.
%   PLOTTER(I) displays the image I. I is either a single
%   multidimensional image or a cell array of images.
%
%   P = PLOTTER(...) returns class object. Exposes vector containing
%   the figure handles as well as a cell array containing the axes handles
%   for each figure. See 'Parameters' below for list of properties that
%   that can be modified.
%
%   PLOTTER(I, ..., PARAM1, VAL1, ...) displays the image I with the
%   defined parameters.
%
%   Parameters include:
%
%   'slice'         Nx1 or 1xN vector or cell array of vectors containing
%                   the slices to display. Default is ceil(size(I)/2).
%
%   'contrast'      [LOW, HIGH] or cell array of vectors specifying display
%                   range of I. Default is [min(I(:)) max(I(:))].
%
%   'kspace'        Binary vector or scalar. 0 equals image space. 1 equals
%                   kspace. Default is image space.
%
%   'subsize'       [ROWS COLUMNS] arrangement for multiple images. Default
%                   is [floor(sqrt(N)) ceil(N/floor(sqrt(N)))]. Where N is
%                   the number of images.
%
%   'color'         String or cell array of strings of colormaps. Default
%                   is 'gray'. To see available colormaps: HELP GRAPH3D
%
%   'scale'         [LOW HIGH] scales all input images to given range.
%
%
%   Features:
%       - 'k'/'K' toggle selected/all images to kspace/images space.
%       - 's'/'S' print current slice of selected/all images.
%       - 'c'/'C' print contrast of selected/all images.
%       - 'upparrow'/'downarrow'/'scrollwheel' change slices.
%
%   See also IMSHOW, FIGURE, AXES

    properties (Access = public)
        hfig
        haxes
    end % public properties


    properties (Dependent)
        kspace
        slice
        contrast
        colors
    end % public dependent properties


    properties (Hidden, Access = public)
        kspace_   = [];
        slice_    = {};
        contrast_ = {};
        colors_   = {};

        haxesDict_
        imgs_
        subSize_

        N_
        siz_
        sizC_
        is2d_
    end % private properties


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Public Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Access = public)

        function [this] = Plotter(x, varargin) % ************************ %
            narginchk(1, 16)

            args = this.parseInputs(x, varargin{:});

            this.imgs_    = args.x;
            this.N_       = args.N;
            this.siz_     = args.siz1;
            this.sizC_    = args.sizC;
            this.subSize_ = args.subSize;

            this.slice    = args.slices;
            this.kspace   = args.kspace;
            this.contrast = args.contrasts;
            this.colors   = args.colors;

            this.hfig     = this.figPlotter;
            [this.haxes, this.haxesDict_] = this.axsPlotter;

            this.PlotterMeThis();
            if nargout == 0, clear('this'); end

        end % Plotter

    end % public methods


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Getter/Setter Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        function set.kspace(this, k) % ********************************** %
            if isempty(this.kspace_)
                this.kspace_ = k;
                return;
            end
            idx = find(this.kspace_(:) ~= k(:)); %#ok<*FNDSB>
            idx = reshape(idx, [1, length(idx)]);
            for ii = idx
                if k(ii) && isempty(this.imgs_{2,ii})
                    this.imgs_{2,ii} = this.fftDisp(this.imgs_{1,ii});
                    this.contrast_{2,ii} = ...
                        this.imLims(this.imgs_{2,ii}, this.slice{ii});
                end
            end
            this.kspace_ = k;
            this.PlotterMeThis(idx);
        end % set kspace

        function [x] = get.kspace(this)
            x = this.kspace_;
        end % get kspace

        function set.slice(this, s) % *********************************** %
            if isempty(this.slice_)
                this.slice_ = s;
                return;
            end
            a = find(cellfun(@(x, y) ~isequal(x, y), this.slice_, s));
            if isempty(a), return; end
            f = find(this.slice_{a} ~= s{a});
            this.slice_ = s;
            this.PlotterMeThis(a, 4-f);
        end % set slice

        function [s] = get.slice(this)
            s = this.slice_;
        end % get slice

        function set.contrast(this, c) % ******************************** %
            if isempty(this.contrast_)
                this.contrast_ = c;
                return;
            end
            if ~iscell(c), c = {c}; end
            this.contrast_ = this.prepContrast(c, this.contrast, ...
                this.kspace_, this.slice_, this.imgs_);
            this.PlotterMeThis();
        end % set contrast

        function [c] = get.contrast(this)
            c = this.contrast_;
        end % get contrast

        function set.colors(this, c) % ********************************** %
            if isempty(this.colors_)
                this.colors_ = c;
                return;
            end

            if ~iscell(c), c = {c}; end
            this.colors_ = this.prepColor(c, this.colors_);
            this.PlotterMeThis();
        end % set colors

        function [c] = get.colors(this)
            c = this.colors_;
        end % get colors

    end % Getter/Setter methods


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Private Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Hidden, Access = private)

        function PlotterMeThis(this, idx, ss) % ************************* %
            if nargin < 3, ss  = [1, 2, 3]; end
            if nargin < 2, idx = 1:this.N_; end

            is2d = 0;
            for ii = 1:idx(1)-1
                if ismatrix(this.imgs_{this.kspace(ii)+1,ii})
                    is2d = is2d + 1;
                end
            end

            for ii = idx
                xx = this.imgs_{this.kspace(ii)+1,ii};
                sl = this.slice{ii};
                li = this.contrast{this.kspace(ii)+1, ii};
                co = this.colors{ii};

                if any(ismember(1, ss))
                    hh = this.haxes{1};
                    this.imShow(imrotate(squeeze(xx(:,:,sl(3))), -90), ...
                        li(1,:), co, hh{ii}); axis off
                end

                if ismatrix(xx)
                    is2d = is2d + 1;
                else
                    if any(ismember(2, ss))
                        hh = this.haxes{2};
                        this.imShow(imrotate(squeeze(xx(:,sl(2),:)), 0), ...
                            li(2,:), co, hh{ii-is2d}); axis off
                    end

                    if any(ismember(3, ss))
                        hh = this.haxes{3};
                        this.imShow(imrotate(squeeze(xx(sl(1),:,:)), 0), ...
                            li(3,:), co, hh{ii-is2d}); axis off
                    end
                end
            end

            if strcmpi(this.hfig{1}.Visible, 'off')
                this.hfig{1}.Visible = 'on';
                if length(idx) == this.N_ && is2d ~= this.N_
                    this.hfig{2}.Visible = 'on';
                    this.hfig{3}.Visible = 'on';
                elseif numel(this.hfig) > 1
                    % delete(this.hfig{2:3});
                    this.hfig(2:3) = [];
                end
            end
        end % PlotterMeThis

        function [h] = figPlotter(this) % ******************************* %
            h = cell(3, 1);
            v = version;
            if str2double(v(3)) < 4
                c = 'black';
            else
                c = 'none';
            end
            for ii = 1:3
                h{ii} = figure( ...
                    'Visible', 'off', ...
                    'Name', 'Plotter Me This', ...
                    'Numbertitle', 'off', ...
                    'Color', c, ...
                    'PaperUnits', 'inches', ...
                    'PaperPositionMode', 'auto', ...
                    'PaperPosition', [0, 0, 6, 3], ...
                    'InvertHardcopy', 'off', ...
                    'PaperSize', [6, 3], ...
                    'WindowStyle', 'docked', ...
                    'Interruptible', 'off', ...
                    'WindowKeyPressFcn', @this.keyPressFcn, ...
                    'WindowScrollWheelFcn', @this.scrollFcn, ...
                    'UserData', ii);

                    set(0, 'showhidden', 'on');
                    ch = get(h{ii}, 'children');
                    chtags = get(ch, 'Tag');
                    ftb_ind = strcmp(chtags, 'FigureToolBar');
                    UT = get(ch(ftb_ind), 'children');
                    try
                        delete(UT((end-4):end));
                        delete(UT(end-10));
                        delete(UT(end-11));
                        delete(UT(end-13));
                        delete(UT(end-14));
                        delete(UT(end-15));
                    catch
                        % void
                    end
                    set(0, 'showhidden', 'off');
            end
        end % figPlotter

        function [hh, hi] = axsPlotter(this) % ************************** %
            hh  = cell(3, 1);
            hi  = cell(3, 1);
            for kk = 1:3
                r   = this.subSize_{kk}(1);
                c   = this.subSize_{kk}(2);
                ir  = 1/r;
                ic  = 1/c;

                axl =   0 :  ic : 1-ic;
                axb = 1-ir : -ir : 0;
                h   = cell(r * c, 1);
                ll  = 1;
                for ii = 1:r
                    for jj = 1:c
                        hi{kk, ll} = [axl(jj), axb(ii), ic, ir];
                        h{ll} = axes( ...
                            'Parent', this.hfig{kk}, ...
                            'Units', 'normalized', ...
                            'Position', hi{kk, ll});
                        ll = ll + 1;
                    end
                end
                hh{kk} = h;
            end
        end % axes Plotter

        function [] = scrollFcn(this, obj, arg)  % ********************** %
            if arg.VerticalScrollCount ~= 0
                this.slice = this.changeSlice(obj, arg.VerticalScrollCount);
            end
        end % scrollFcn

        function [] = keyPressFcn(this, obj, arg)  % ******************** %
            switch arg.Key
                case 'c'
                    if strcmp(arg.Character, arg.Key)
                        f = obj.UserData;
                        idx = this.axesIdx(obj.CurrentAxes.Position, ...
                            this.haxesDict_(f,:));
                        this.printContrast(idx);
                    else
                        this.printContrast();
                    end

                case 'k'
                    if strcmp(arg.Character, arg.Key)
                        f = obj.UserData;
                        idx = this.axesIdx(obj.CurrentAxes.Position, ...
                            this.haxesDict_(f,:));
                        tmp = this.kspace;
                        tmp(idx) = xor(tmp(idx), 1);
                        this.kspace = tmp;
                    else
                        this.kspace = xor(this.kspace, 1);
                    end

                case 's'
                    if strcmp(arg.Character, arg.Key)
                        f = obj.UserData;
                        idx = this.axesIdx(obj.CurrentAxes.Position, ...
                            this.haxesDict_(f,:));
                        this.printSlice(idx);
                    else
                        this.printSlice();
                    end

                case 'uparrow'
                    this.slice = this.changeSlice(obj, +1);

                case 'downarrow'
                    this.slice = this.changeSlice(obj, -1);
            end
        end % keyPressFcn


        function [s] = changeSlice(this, obj, val) % ******************** %
            f = obj.UserData;
            a = this.axesIdx(obj.CurrentAxes.Position, this.haxesDict_(f,:));
            s = this.slice;
            tmp = s{a}(4-f) + val;
            s{a}(4-f) = max(min(tmp, this.siz_{a}(4-f)), 1);
        end % changeSlice

        function printContrast(this, idx) % ***************************** %
            if nargin < 2
                idx = 1:this.N_;
            end
            for ii = idx
                fprintf('Image: %d :: Contrast:', ii);
                fprintf('  [%1.2f, %1.2f]', this.contrast{this.kspace(ii)+1,ii}');
                fprintf('\n');
            end
            fprintf('\n');
        end % printContrast

        function printSlice(this, idx) % ******************************** %
            if nargin < 2
                idx = 1:this.N_;
            end
            for ii = idx
                fprintf('Image: %d :: Slice:', ii);
                fprintf(' [%d, %d, %d]', this.slice{ii}+this.sizC_{ii})
                fprintf('\n');
            end
            fprintf('\n');
        end % printSlice

    end % private methods


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Static Private Methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Static, Hidden, Access = private)

        function imShow(x, lims, colors, h) % *************************** %
            imshow(...
                imrotate(x, 90), lims, ...
                'Border', 'tight', ...
                'Colormap', colormap(colors), ...
                'InitialMagnification', 'fit', ...
                'Parent', h);
        end % imShow

        function [idx] = axesIdx(pos, dict) % *************************** %
            for ii = 1:numel(dict)
                if abs(pos - dict{ii}) < eps('single')
                    break;
                end
            end
            idx = ii;
        end % axesIdx

        function [X] = fftDisp(x) % ************************************* %
            X = log(1 + abs(fftshift(fftn(x))));
        end % fftDisp

    end % static private methods


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Input Parsing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Static, Hidden, Access = private)

        function [args] = parseInputs(x, varargin) % ******************** %
            if ~iscell(x), x = {x}; end
            n  = length(x);
            nn = 0;

            for ii = 1:n
                validateattributes(x{ii}, {'single', 'double', 'logical'}, ...
                    {'nonnan', 'nonempty', 'nonsparse', 'finite'}, ...
                    mfilename, 'imgs', 1);
                nn = nn + size(x{ii}, 4);
            end

            [tmp, siz0, siz1, sizC] = deal(cell(1, nn));
            idx  = 1;
            is2d = 0;

            for ii = 1:n
               for jj = 1:size(x{ii}, 4)
                   if ~isreal(x{ii})
                       t = abs(x{ii}(:,:,:,jj));
                   else
                       t = x{ii}(:,:,:,jj);
                   end

                   [m, n, p] = size(t);
                   siz0{idx} = [m, n, p];
                   [tmp{idx}, sizC{idx}] = Plotter.cropX(t);
                   [m, n, p] = size(tmp{idx});
                   siz1{idx} = [m, n, p];

                   if p == 1, is2d = is2d +1; end
                   idx = idx + 1;
               end
            end

            n = nn;
            x = tmp;
            clear tmp t

            dSlice  = cellfun(@(x) ceil(x/2), siz1, 'UniformOutput', false);
            dKspace = zeros(n, 1);
            dColors = repmat({'gray'}, n, 1);
            dSubSize(1) = floor(sqrt(n));
            dSubSize(2) = ceil(n / dSubSize(1));

            p = inputParser;
            p.FunctionName = mfilename;

            p.addParamValue('contrast', 0, @(x)validateattributes(x, ...
                {'cell', 'double'}, {'nonempty'}));

            p.addParamValue('scale', 0, @(x)validateattributes(x, ...
                {'double'}, {'vector', 'nonempty', 'increasing'}));

            p.addParamValue('slice', dSlice, @(x)validateattributes(x, ...
                {'cell', 'double'}, {'nonempty'}));

            p.addParamValue('kspace', dKspace, @(x)validateattributes(x, ...
                {'numeric', 'logical'}, {'vector', 'nonempty'}));

            p.addParamValue('subsize', dSubSize, @(x)validateattributes(x, ...
                {'double'}, {'vector', 'nonempty', 'nonnegative'}));

            p.addParamValue('color', dColors, @(x)validateattributes(x, ...
                {'cell', 'char'}, {'nonempty'}));

            p.parse(varargin{:})

            if ~iscell(p.Results.slice)
                pSlices = {p.Results.slice};
            else
                pSlices = p.Results.slice;
            end

            if ~iscell(p.Results.color)
                pColormaps = {p.Results.color};
            else
                pColormaps = p.Results.color;
            end

            if ~iscell(p.Results.contrast)
                pContrasts = {p.Results.contrast};
            else
                pContrasts = p.Results.contrast;
            end

            if ~isscalar(p.Results.scale)
                x = cellfun(@(x) Plotter.scaleX(x, p.Results.scale), x, ...
                    'UniformOutput', false);
            end

            if is2d > 0 && is2d ~= n
                pSubSize{1} = p.Results.subsize;
                pSubSize{2}(1) = floor(sqrt(n-is2d));
                pSubSize{2}(2) = ceil((n-is2d) / dSubSize(1));
                pSubSize{3} = pSubSize{2};
            else
                [pSubSize{1:3}] = deal(p.Results.subsize);
            end

            args.kspace = Plotter.prepKspace(p.Results.kspace, dKspace);

            X = cell(1, n);
            for ii = 1:n
                if args.kspace(ii)
                    X{ii} = Plotter.fftDisp(x{ii});
                end
            end

            args.x = [x; X];
            args.N = n;
            args.siz1 = siz1;
            args.sizC = sizC;

            args.slices    = Plotter.prepSlice(pSlices, siz1, sizC, dSlice);
            args.colors    = Plotter.prepColor(pColormaps, dColors);
            args.subSize   = pSubSize;
            args.contrasts = Plotter.prepContrast(pContrasts, [], ...
                args.kspace, args.slices, args.x);
        end % parseInputs

        function [x, offset] = cropX(x) % ******************************* %
            offset = [0, 0, 0];
            %tmp = logical(x);

            %if ~any(tmp(:)), return; end

            %idx = max(max(tmp, [], 2), [], 3);
            %idy = max(max(tmp, [], 1), [], 3);
            %idz = max(max(tmp, [], 1), [], 2);

            %mi = min(x(:));
            %x(~tmp) = round(1000 * mi * (-(mi > 0) + (mi < 0)));
            %x = x(idx, idy, idz);
            %offset = [find(idx, 1), find(idy, 1), find(idz, 1)];
        end % cropX

        function [x] = scaleX(x, s) % *********************************** %
            mi = min(x(:));
            ma = max(x(:));
            ss = s(2) - s(1);
            x  = ss * (x - mi)/(ma - mi) + s(1);
        end % scaleX

        function [cc] = prepContrast(c0, d, kspace, slices, x) % ******** %
            if isempty(d)
                cc = cell(2, size(x, 2));
                for ii = 1:size(x, 2)
                    s  = slices{ii};
                    X  = x{1,ii};
                    cc{1,ii} = Plotter.imLims(X, s);

                    if kspace(ii)
                        X = x{2,ii};
                        cc{2,ii} = Plotter.imLims(X, s);
                    end
                end
            else
                cc = d;
            end

            if ~isscalar(c0{1})
                n = length(c0);
                for ii = 1:size(x, 2)
                    if ii > n
                        t = c0{n};
                    else
                        t = c0{ii};
                    end

                    m = size(t, 1);
                    for jj = 1:3
                        if jj > m
                            tt = t(m,:);
                        else
                            tt = t(jj,:);
                        end
                        if tt(1) < tt(2)
                            cc{kspace(ii)+1,ii}(jj,:) = double(tt);
                        end
                    end
                end
            end
        end % prepContrasts

        function [s1] = prepSlice(s0, sz, offset, d) % ***************** %
            s1 = d;
            if isequal(s0, d), return; end

            n  = length(s0);
            for ii = 1:length(d)
                t = ones(1, 3);
                if ii > n
                    t(1:length(s0{n})) = s0{n};
                else
                    t(1:length(s0{ii})) = s0{ii};
                end
                t = t - offset{ii};
                if all(t < sz{ii})
                    s1{ii} = max(t, 1);
                end
            end
        end % prepSlices

        function [k1] = prepKspace(k0, d) % ***************************** %
            k1 = d;
            if isequal(k0, d), return; end

            n = length(k0);
            for ii = 1:length(d)
                if ii > n
                    t = k0(n);
                else
                    t = k0(ii);
                end
                k1(ii) = t;
            end
        end % prepKspace

        function [c1] = prepColor(c0, d) % ****************************** %
            c1 = d;
            if isequal(c0, d), return; end
            h = figure('Visible', 'off');
            n = length(c0);
            for ii = 1:length(d)
                if ii > n
                    t = c0{n};
                else
                    t = c0{ii};
                end
                try
                    colormap(t);
                    c1{ii} = t;
                catch %#ok<CTCH>
                    warning('Colormap: %s does NOT exist!', t);
                end
            end
            delete(h);
        end % prepColors

        function [li] = imLims(x, s)
            XX = squeeze(x(s(1),:,:));
            c1 = [min(XX(:)), max(XX(:))];

            XX = squeeze(x(:,s(2),:));
            c2 = [min(XX(:)), max(XX(:))];

            XX = squeeze(x(:,:,s(3)));
            c3 = [min(XX(:)), max(XX(:))];

            li = double([c1; c2; c3]);
        end

    end % Input parsing


end % classdef
