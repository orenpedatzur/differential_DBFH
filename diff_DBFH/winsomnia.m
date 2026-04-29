function winsomnia(enable_bool)
    
    %winsomnia - disable sleep in Windows
    %
    % winsomnia(true) creates a duplicate of the currently active power plan, adds
    % '+winsomnia' to its name and sets all its powersaving sleep timers (screen,
    % computer, hibernation, disk) to 'never'.
    %
    % Use winsomnia(false) to restore the previous power plan. For example, if the
    % current winsomnia plan is called 'Balanced+winsomnia', the call will
    % reactivate the plan named 'Balanced'. In the event the 'Balanced' plan has
    % been deleted since 'Balanced+winsomnia' was created, a warning with ID
    % 'winsomnia:cant_revert' is thrown and the winsomnia plan remains active.
    %
    % Calling winsomnia(true) when the current power plan name contains '+winsomnia'
    % throws a warning with ID 'winsomnia:already_insomniac' and returns without
    % changing anything.
    %
    % Calling winsomnia(false) when no power plan name contains '+winsomnia' quietly
    % returns without changing anything. This is means it's safe to call
    % winsomnia(false) whenever, no need to keep track.
    %
    % This function is based on the Windows command 'powercfg' which is documented here:
    % https://docs.microsoft.com/en-us/windows-hardware/design/device-experiences/powercfg-command-line-options
    %
    % Example:
    %   winsomnia(true)
    %   % <your script>
    %   winsomnia(false)
    %
    % Tips:
    %   Type "!powercfg list" on the Matlab commandline to see the list of power
    %   plans available on your computer with an asterisk indicating the currently
    %   active one.
    %
    %   To see the Power Plan Control Panel, type "!control powercfg.cpl" on the
    %   Matlab commandline.
    %
    %   The Power Plan Control Panel needs to be closed and re-opened for the
    %   effects of this function to become fully visible in it.
    %
    %   Add winsomia(false) to your "finish.m" to reset the power plan automatically
    %   upon exiting Matlab.
    %
    % See also: <a href="matlab:system('control powercfg.cpl');">control powercfg.cpl</a>
    
    % Jacob Duijnhouwer 2020-10-27
    
    %%
    if ~ispc
        error('winsomnia only works on Windows');
    end
    
    %%
    if nargin~=1 || ~islogical(enable_bool)
        error('Input must be true of false');
    end
  
    %%
    if enable_bool
        %% create and enable the winsomnia plan
        % Get the GUID of the current power plan so we can revert to it later
        cmd = 'powercfg getactivescheme';
        msg= string(mysysdo(cmd));
        old_powerplan_name = extractBetween(msg,'(',')');
        if contains(old_powerplan_name,'+winsomnia')
            warning('winsomnia:already_insomniac','A power plan with a name containing "+winsomnia" is already active, no changes were made.');
            return
        end
        old_powerplan_guid = strtrim(extractBetween(msg,'GUID:','('));
        
        % Make a copy of the currently active power plan
        cmd=sprintf('powercfg duplicatescheme %s',old_powerplan_guid);
        msg= string(mysysdo(cmd));
        new_powerplan_guid = strtrim(extractBetween(msg,'GUID:','('));
        
        % Rename the new power plan (to the name of this function) to and add a description
        cmd=sprintf('powercfg changename %s "%s+winsomnia" "Programmatically created with MATLAB:winsomnia.m on %s"',new_powerplan_guid,old_powerplan_name,datestr(now));
        mysysdo(cmd);
        
        % Activate the new power plan
        mysysdo(sprintf('powercfg setactive %s',new_powerplan_guid));
        
        % Turn off all sleep options in the currently active plan
        mysysdo('powercfg change -hibernate-timeout-ac 0');
        mysysdo('powercfg change -hibernate-timeout-dc 0');
        mysysdo('powercfg change -disk-timeout-ac 0');
        mysysdo('powercfg change -disk-timeout-dc 0');
        mysysdo('powercfg change -monitor-timeout-ac 0');
        mysysdo('powercfg change -monitor-timeout-dc 0');
        mysysdo('powercfg change -standby-timeout-ac 0');
        mysysdo('powercfg change -standby-timeout-dc 0');      
    else
        %% create and enable the winsomnia plan
        %  Make the original power plan active again and delete the winsomnia plan
        msg=mysysdo('powercfg list');
        msg(isspace(msg))=' '; % The list output contains all sorts of funky white space characters, replace with plain spaces
        msg=string(msg);
        plan_lines = extractBetween(msg,'GUID: ',')');
        % find the plan_line that contains +winsomnia
        idx=contains(plan_lines,'+winsomnia');
        if ~any(idx)
            % Calling winsomnia(false) when no power plan contains '+winsomnia' quietly
            % returns without changing anything.
            return
        end
        winsomnia_line=plan_lines(idx);
        % what are the winsomnia GUID and name?
        winsomnia_guid = strtrim(extractBefore(string(winsomnia_line),'('));
        winsomnia_name = extractAfter(string(winsomnia_line),'(');
        % what is the name of the original plan the +winsomnia plan was based on?
        original_name=extractBefore(winsomnia_name,'+winsomnia');
        % Find the GUID of the original plan
        original_guid = strtrim(extractBefore(plan_lines(endsWith(plan_lines,original_name)),'('));
        if isempty(original_guid)
            warning('winsomnia:cant_revert','No power plan named "%s" on this computer. "%s" remains active.',original_name,winsomnia_name);
            return
        end
        % Find the GUID of the currently active plan (does not have to be the winsomnia
        % one in case it got changed since the winsomnia(true) call
        msg=mysysdo('powercfg getactivescheme');
        msg=string(msg);
        active_guid = strtrim(extractBetween(msg,'GUID:','('));
        % If the winsomnia is active, make the original plan active
        if strcmpi(active_guid,winsomnia_guid) 
            mysysdo(sprintf('powercfg setactive %s',original_guid));
        end
        % Delete all the plans whose names contain '+winsomnia'. Except in the unlikely
        % but not impossible event that all plans contain "+winsomnia", the delete all
        % but one and throw a warning. As it stands it's actually impossible that all
        % plans contain "+winsomnia" because this function would already have thrown a
        % "cant_revert" warning and returned above. Just leaving it here out of an
        % abundance of caution, it's probably not a bad idea to delete all power plans
        % from the computer.
        n_deleted=0;
        for i=1:numel(plan_lines)
            if contains(plan_lines{i},'+winsomnia')
                winsomnia_guid = strtrim(extractBefore(string(plan_lines(i)),'('));
                if n_deleted==numel(plan_lines)-1
                    warning('winsomnia:all_winsomnia','The names of all power plans contained the string "+winsomnia". Deleted all power plans but the last one.');
                    return
                else
                    mysysdo(sprintf('powercfg delete %s',winsomnia_guid));
                    n_deleted=n_deleted+1;
                end
            end
        end
    end
    
    %% Wrapper around the system function that catches errors and attempts to revert to original powerplan in case of errors.
    function msg=mysysdo(cmd)
        [err,msg]=system(cmd);
        if err
            % Try resetting the old_power_scheme
            if exist('old_powerplan_guid','var') && ~isempty(old_powerplan_guid)
                system(sprintf('powercfg /setactive %s',old_powerplan_guid));
            end
            !powercfg list
            error('System call ''%s'' returned an error:\n\t"%s"',cmd,msg);
        end
    end
    
end






