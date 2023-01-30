import os

# Command constants.
# WEIGHT_LEARN_COMMAND = "WEIGHTLEARN"
WEIGHT_LEARN_COMMAND = ""
WRITE_INFERRED_COMMAND = "WRITEINFERREDPREDICATES"
ADD = 'ADDATOM'
OBSERVE = 'OBSERVEATOM'
UPDATE = 'UPDATEATOM'
DELETE = 'DELETEATOM'
FIX = 'FIXATOM'
CLOSE_COMMAND = 'STOP'
EXIT_COMMAND = 'EXIT'

# Partition names
OBS = 'obs'
TRUTH = 'truth'
TARGET = 'target'


def incremental_construct_client_commands(predicate_constructor,
                                          observed_departure_demands_df, observed_arrival_demands_df,
                                          new_target_departure_demands_df, new_target_arrival_demands_df,
                                          expired_departure_demands_df, expired_arrival_demands_df,
                                          new_time_block_df, expired_time_block_df,
                                          time_step):
    command_list = []
    add_observation_command_list = []
    observe_command_list = []
    delete_observation_command_list = []
    delete_time_block_command_list = []

    write_inferred_commands = [WRITE_INFERRED_COMMAND + "\t'./inferred-predicates/{:03d}/{:02d}'".format(time_step, i) for i in range(24)]

    if observed_departure_demands_df is not None:
        new_observed_departure_demands_df = observed_departure_demands_df
        command_list += df_to_command(
        predicate_constructor.time_to_int_ids(new_observed_departure_demands_df.loc[:, ['station_id', 'time']]),
        new_observed_departure_demands_df.loc[:, ['demand']],
        ADD, TRUTH, 'DepartureDemand')

    if observed_arrival_demands_df is not None:
        new_observed_arrival_demands_df = observed_arrival_demands_df
        command_list += df_to_command(
        predicate_constructor.time_to_int_ids(new_observed_arrival_demands_df.loc[:, ['station_id', 'time']]),
        new_observed_arrival_demands_df.loc[:, ['demand']],
        ADD, TRUTH, 'ArrivalDemand')

    command_list += [WEIGHT_LEARN_COMMAND]

    if observed_departure_demands_df is not None:
        new_observed_departure_demands_df = observed_departure_demands_df
        observe_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(new_observed_departure_demands_df.loc[:, ['station_id', 'time']]),
            new_observed_departure_demands_df.loc[:, ['demand']],
            UPDATE, OBS, 'DepartureDemand')

    if observed_arrival_demands_df is not None:
        new_observed_arrival_demands_df = observed_arrival_demands_df
        observe_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(new_observed_arrival_demands_df.loc[:, ['station_id', 'time']]),
            new_observed_arrival_demands_df.loc[:, ['demand']],
            UPDATE, OBS, 'ArrivalDemand')

    command_list += observe_command_list
    command_list += add_observation_command_list

    if expired_departure_demands_df is not None:
        delete_observation_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(expired_departure_demands_df.loc[:, ['station_id', 'time']]),
            expired_departure_demands_df.loc[:, ['demand']],
            DELETE, OBS, 'DepartureDemand')

    if expired_arrival_demands_df is not None:
        delete_observation_command_list += df_to_command(
            predicate_constructor.time_to_int_ids(expired_arrival_demands_df.loc[:, ['station_id', 'time']]),
            expired_arrival_demands_df.loc[:, ['demand']],
            DELETE, OBS, 'ArrivalDemand')

    if expired_time_block_df is not None:
            delete_time_block_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(expired_time_block_df.loc[:, ['station_id', 'time']]),
                expired_time_block_df.loc[:, ['value']],
                DELETE, OBS, 'TimeBlock')

    command_list += delete_observation_command_list + delete_time_block_command_list

    hour_add_targets_command_list = [[] for _ in range(24)]
    hour_add_time_block_command_list = [[] for _ in range(24)]
    hour_fix_inferred_targets_command_list = [[] for _ in range(24)]
    hour_infer_predicates_command_list = [[command] for command in write_inferred_commands]

    for hour in range(24):
        if new_time_block_df is not None:
            hour_add_time_block_command_list[hour] += df_to_command(
                predicate_constructor.time_to_int_ids(new_time_block_df.loc[new_time_block_df.time.dt.hour == hour, ['station_id', 'time']]),
                new_time_block_df.loc[new_time_block_df.time.dt.hour == hour, ['value']],
                ADD, OBS, 'TimeBlock')

        # DEMAND.
        if new_target_departure_demands_df is not None:
            new_departure_targets_df = new_target_departure_demands_df.reset_index()
            new_departure_targets_df = new_departure_targets_df.loc[new_departure_targets_df.time.dt.hour == hour]
            hour_add_targets_command_list[hour] += df_to_command(
                predicate_constructor.time_to_int_ids(new_departure_targets_df.loc[:, ['station_id', 'time']]),
                new_departure_targets_df.loc[:, []],
                ADD, TARGET, 'DepartureDemand')
            hour_fix_inferred_targets_command_list[hour] += df_to_command(
                predicate_constructor.time_to_int_ids(new_departure_targets_df.loc[:, ['station_id', 'time']]),
                new_departure_targets_df.loc[:, []],
                FIX, OBS, 'DepartureDemand')

        if new_target_arrival_demands_df is not None:
            new_arrival_targets_df = new_target_arrival_demands_df.reset_index()
            new_arrival_targets_df = new_arrival_targets_df.loc[new_arrival_targets_df.time.dt.hour == hour]
            hour_add_targets_command_list[hour] += df_to_command(
                predicate_constructor.time_to_int_ids(new_arrival_targets_df.loc[:, ['station_id', 'time']]),
                new_arrival_targets_df.loc[:, []],
                ADD, TARGET, 'ArrivalDemand')
            hour_fix_inferred_targets_command_list[hour] += df_to_command(
                predicate_constructor.time_to_int_ids(new_arrival_targets_df.loc[:, ['station_id', 'time']]),
                new_arrival_targets_df.loc[:, []],
                FIX, OBS, 'ArrivalDemand')

    for hour in range(24):
        command_list += hour_add_targets_command_list[hour] \
                        + hour_add_time_block_command_list[hour] \
                        + hour_infer_predicates_command_list[hour] \
                        + hour_fix_inferred_targets_command_list[hour]

    return (command_list)


def online_construct_client_commands(predicate_constructor,
                                     observed_departure_demands_df, observed_arrival_demands_df,
                                     new_target_departure_demands_df, new_target_arrival_demands_df,
                                     expired_departure_demands_df, expired_arrival_demands_df,
                                     new_time_block_df, expired_time_block_df,
                                     time_step):
    add_targets_command_list = []
    add_time_block_command_list = []
    delete_time_block_command_list = []
    add_observation_command_list = []
    delete_observation_command_list = []
    observe_command_list = []
    weight_learning_commands = []
    write_inferred_commands = [WRITE_INFERRED_COMMAND + "\t'./inferred-predicates/{:03d}'".format(time_step)]

    if observed_departure_demands_df is not None:
        new_observed_departure_demands_df = observed_departure_demands_df
        weight_learning_commands += df_to_command(
            predicate_constructor.time_to_int_ids(new_observed_departure_demands_df.loc[:, ['station_id', 'time']]),
            new_observed_departure_demands_df.loc[:, ['demand']],
            ADD, TRUTH, 'DepartureDemand')

    if observed_arrival_demands_df is not None:
        new_observed_arrival_demands_df = observed_arrival_demands_df
        weight_learning_commands += df_to_command(
            predicate_constructor.time_to_int_ids(new_observed_arrival_demands_df.loc[:, ['station_id', 'time']]),
            new_observed_arrival_demands_df.loc[:, ['demand']],
            ADD, TRUTH, 'ArrivalDemand')

    weight_learning_commands += [WEIGHT_LEARN_COMMAND]

    if time_step > 0:
        # DEMAND.
        if new_target_departure_demands_df is not None:
            new_departure_targets_df = new_target_departure_demands_df.reset_index()
            add_targets_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(new_departure_targets_df.loc[:, ['station_id', 'time']]),
                new_departure_targets_df.loc[:, []],
                ADD, TARGET, 'DepartureDemand')

        if observed_departure_demands_df is not None:
            new_observed_departure_demands_df = observed_departure_demands_df
            observe_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(new_observed_departure_demands_df.loc[:, ['station_id', 'time']]),
                new_observed_departure_demands_df.loc[:, ['demand']],
                OBSERVE, OBS, 'DepartureDemand')

        if expired_departure_demands_df is not None:
            delete_observation_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(expired_departure_demands_df.loc[:, ['station_id', 'time']]),
                expired_departure_demands_df.loc[:, ['demand']],
                DELETE, OBS, 'DepartureDemand')

        if new_target_arrival_demands_df is not None:
            new_arrival_targets_df = new_target_arrival_demands_df.reset_index()
            add_targets_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(new_arrival_targets_df.loc[:, ['station_id', 'time']]),
                new_arrival_targets_df.loc[:, []],
                ADD, TARGET, 'ArrivalDemand')

        if observed_arrival_demands_df is not None:
            new_observed_arrival_demands_df = observed_arrival_demands_df
            observe_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(new_observed_arrival_demands_df.loc[:, ['station_id', 'time']]),
                new_observed_arrival_demands_df.loc[:, ['demand']],
                OBSERVE, OBS, 'ArrivalDemand')

        if expired_arrival_demands_df is not None:
            delete_observation_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(expired_arrival_demands_df.loc[:, ['station_id', 'time']]),
                expired_arrival_demands_df.loc[:, ['demand']],
                DELETE, OBS, 'ArrivalDemand')

        # TIME BLOCK
        if new_time_block_df is not None:
            add_time_block_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(new_time_block_df.loc[:, ['station_id', 'time']]),
                new_time_block_df.loc[:, ['value']],
                ADD, OBS, 'TimeBlock')

        if expired_time_block_df is not None:
            delete_time_block_command_list += df_to_command(
                predicate_constructor.time_to_int_ids(expired_time_block_df.loc[:, ['station_id', 'time']]),
                expired_time_block_df.loc[:, ['value']],
                DELETE, OBS, 'TimeBlock')

    command_list = (weight_learning_commands +
                    observe_command_list + add_targets_command_list +
                    add_observation_command_list + add_time_block_command_list +
                    delete_observation_command_list + delete_time_block_command_list +
                    write_inferred_commands)

    return command_list


def command_file_write(command_list, path, experiment):
    command_str = ''
    for command in command_list:
        command_str += command + '\n'

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, '{}_commands.txt'.format(experiment)), 'w') as writer:
        writer.write(command_str)


def df_to_command(constants_df, value_series, action_type, partition_name, predicate_name):
    command_list = []
    assert (constants_df.shape[0] == value_series.shape[0])

    for idx, row in constants_df.iterrows():
        predicate_constants = row.values
        if value_series.loc[idx].shape[0] != 0:
            value = value_series.loc[idx].values[0]
        else:
            value = None
        command_list += [create_command_line(action_type, partition_name, predicate_name, predicate_constants, value)]
    return command_list


def create_command_line(action_type, partition_name, predicate_name, predicate_constants, value):
    if partition_name == OBS:
        partition_str = "READ"
    elif partition_name == TARGET:
        partition_str = "WRITE"
    elif partition_name == TRUTH:
        partition_str = "TRUTH"

    quoted_predicate_constants = ["'" + str(const) + "'" for const in predicate_constants]
    constants_list = ",".join(quoted_predicate_constants)

    if action_type == ADD:
        if value is not None:
            return ADD + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)
        else:
            return ADD + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")"

    if action_type == FIX:
        return FIX + "\t" + predicate_name + "(" + constants_list + ")\t"

    if action_type == OBSERVE:
        return OBSERVE + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)

    elif action_type == UPDATE:
        return UPDATE + "\t" + predicate_name + "(" + constants_list + ")\t" + str(value)

    elif action_type == DELETE:
        return DELETE + "\t" + partition_str + "\t" + predicate_name + "(" + constants_list + ")"
