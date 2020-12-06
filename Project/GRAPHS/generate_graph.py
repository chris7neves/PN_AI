from SQL import Data_Extract
import matplotlib.pyplot as plt
import numpy as np


class graph_output:
    def speed_vs_trial():
        PNAI_SERVER = "26.105.85.122"
        PNAI_DATABASE = "PNLab"
        PNAI_USERNAME = "adm"
        PNAI_PASSWORD = "massimo123"

        SQLObject = Data_Extract.SQLDatabase(PNAI_SERVER,
                                             PNAI_DATABASE,
                                             PNAI_USERNAME,
                                             PNAI_PASSWORD)

        # Make initial query to find out all excel files in Database
        All_Excel_Files = SQLObject.participants()

        asd_samples_x = SQLObject.query(
            "select trial_num from trial_information where notes like \'ASD\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by trial_num")
        asd_samples_y = SQLObject.query(
            "select avg(average_speed) from trial_information where notes like \'ASD\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by trial_num")

        non_asd_samples_x = SQLObject.query(
            "select trial_num from trial_information where notes like \'TYP\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by trial_num")
        non_asd_samples_y = SQLObject.query(
            "select avg(average_speed) from trial_information where notes like \'TYP\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by trial_num")

        # plt.scatter(asd_samples_x, asd_samples_y, color='r', label=f'ASD = {len(asd_samples_x)}')
        # plt.scatter(non_asd_samples_x, non_asd_samples_y, color='b', label=f'Non-ASD = {len(non_asd_samples_x)}')
        plt.scatter(asd_samples_x, asd_samples_y, color='r', label=f'ASD', marker=8)
        plt.scatter(non_asd_samples_x, non_asd_samples_y, color='b', label=f'Non-ASD', marker=9)
        plt.legend(loc='upper right', frameon=True)
        # naming the x axis
        plt.xlabel('Trial')
        plt.xticks(np.arange(1, 12, step=1))
        # naming the y axis
        plt.ylabel('Average Speed')

        # giving a title to my graph
        plt.title('ASD vs Non-ASD Average Speed vs Trial Eye Tracker Experiment')

        # function to show the plot
        plt.show()

        print("hello")

    def correct_vs_trial():
        PNAI_SERVER = "26.105.85.122"
        PNAI_DATABASE = "PNLab"
        PNAI_USERNAME = "adm"
        PNAI_PASSWORD = "massimo123"

        SQLObject = Data_Extract.SQLDatabase(PNAI_SERVER,
                                             PNAI_DATABASE,
                                             PNAI_USERNAME,
                                             PNAI_PASSWORD)

        # Make initial query to find out all excel files in Database
        All_Excel_Files = SQLObject.participants()

        asd_samples_x = SQLObject.query(
            "select trial_num from trial_information where notes like \'ASD\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by trial_num")
        asd_samples_y = SQLObject.query(
            "select avg(t1+t2+t3+t4+t5+t6+t7+t8) from trial_information where notes like \'ASD\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by trial_num")

        non_asd_samples_x = SQLObject.query(
            "select trial_num from trial_information where notes like \'TYP\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by trial_num")
        non_asd_samples_y = SQLObject.query(
            "select avg(t1+t2+t3+t4+t5+t6+t7+t8) from trial_information where notes like \'TYP\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by trial_num")

        # plt.scatter(asd_samples_x, asd_samples_y, color='r', label=f'ASD = {len(asd_samples_x)}')
        # plt.scatter(non_asd_samples_x, non_asd_samples_y, color='b', label=f'Non-ASD = {len(non_asd_samples_x)}')
        plt.scatter(asd_samples_x, asd_samples_y, color='r', label=f'ASD', marker=8)
        plt.scatter(non_asd_samples_x, non_asd_samples_y, color='b', label=f'Non-ASD', marker=9)
        plt.legend(loc='upper right', frameon=True)
        # naming the x axis
        plt.xlabel('Trial')
        plt.xticks(np.arange(1, 12, step=1))
        # naming the y axis
        plt.ylabel('Average Correct Responses')

        # giving a title to my graph
        plt.title('ASD vs Non-ASD Average Correct Responses vs Trial Eye Tracker Experiment')

        # function to show the plot
        plt.show()

        print("hello")

    def speed_vs_age():
        PNAI_SERVER = "26.105.85.122"
        PNAI_DATABASE = "PNLab"
        PNAI_USERNAME = "adm"
        PNAI_PASSWORD = "massimo123"

        SQLObject = Data_Extract.SQLDatabase(PNAI_SERVER,
                                             PNAI_DATABASE,
                                             PNAI_USERNAME,
                                             PNAI_PASSWORD)

        # Make initial query to find out all excel files in Database
        All_Excel_Files = SQLObject.participants()

        asd_samples_x = SQLObject.query(
            "select max(age) from trial_information where notes like \'ASD\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by t_name")
        asd_samples_y = SQLObject.query(
            "select avg(average_speed) from trial_information where notes like \'ASD\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by t_name")

        non_asd_samples_x = SQLObject.query(
            "select max(age) from trial_information where notes like \'TYP\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by t_name")
        non_asd_samples_y = SQLObject.query(
            "select avg(average_speed) from trial_information where notes like \'TYP\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by t_name")

        # plt.scatter(asd_samples_x, asd_samples_y, color='r', label=f'ASD = {len(asd_samples_x)}')
        # plt.scatter(non_asd_samples_x, non_asd_samples_y, color='b', label=f'Non-ASD = {len(non_asd_samples_x)}')
        plt.scatter(asd_samples_x, asd_samples_y, color='r', label=f'ASD')
        plt.scatter(non_asd_samples_x, non_asd_samples_y, color='b', label=f'Non-ASD')

        asdparr_x = np.array(asd_samples_x)
        asdparr_y = np.array(asd_samples_y)
        z = np.polyfit(asdparr_x.flatten(), asdparr_y.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(asd_samples_x, p(asd_samples_x), "r--")

        non_asdparr_x = np.array(non_asd_samples_x)
        non_asdparr_y = np.array(non_asd_samples_y)
        z = np.polyfit(non_asdparr_x.flatten(), non_asdparr_y.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(non_asd_samples_x, p(non_asd_samples_x), "b--")

        plt.legend(loc='upper right', frameon=True)
        # naming the x axis
        plt.xlabel('Age')
        # naming the y axis
        plt.ylabel('Average Speed')

        # giving a title to my graph
        plt.title('ASD vs Non-ASD Average Speed vs Age Eye Tracker Experiment')

        # function to show the plot
        plt.show()

        print("hello")

    def correct_vs_age():
        PNAI_SERVER = "26.105.85.122"
        PNAI_DATABASE = "PNLab"
        PNAI_USERNAME = "adm"
        PNAI_PASSWORD = "massimo123"

        SQLObject = Data_Extract.SQLDatabase(PNAI_SERVER,
                                             PNAI_DATABASE,
                                             PNAI_USERNAME,
                                             PNAI_PASSWORD)

        # Make initial query to find out all excel files in Database
        All_Excel_Files = SQLObject.participants()

        asd_samples_x = SQLObject.query(
            "select max(age) from trial_information where notes like \'ASD\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by t_name")
        asd_samples_y = SQLObject.query(
            "select sum(t1+t2+t3+t4+t5+t6+t7+t8) from trial_information where notes like \'ASD\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by t_name")

        non_asd_samples_x = SQLObject.query(
            "select max(age) from trial_information where notes like \'TYP\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by t_name")
        non_asd_samples_y = SQLObject.query(
            "select sum(t1+t2+t3+t4+t5+t6+t7+t8) from trial_information where notes like \'TYP\' and t_name not in (\'A008\',\'A011\',\'A026\',\'A028\',\'A034\',\'A006\') group by t_name")

        # plt.scatter(asd_samples_x, asd_samples_y, color='r', label=f'ASD = {len(asd_samples_x)}')
        # plt.scatter(non_asd_samples_x, non_asd_samples_y, color='b', label=f'Non-ASD = {len(non_asd_samples_x)}')
        plt.scatter(asd_samples_x, asd_samples_y, color='r', label=f'ASD')
        plt.scatter(non_asd_samples_x, non_asd_samples_y, color='b', label=f'Non-ASD')

        asdparr_x = np.array(asd_samples_x)
        asdparr_y = np.array(asd_samples_y)
        z = np.polyfit(asdparr_x.flatten(), asdparr_y.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(asd_samples_x, p(asd_samples_x), "r--")

        non_asdparr_x = np.array(non_asd_samples_x)
        non_asdparr_y = np.array(non_asd_samples_y)
        z = np.polyfit(non_asdparr_x.flatten(), non_asdparr_y.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(non_asd_samples_x, p(non_asd_samples_x), "b--")


        plt.legend(loc='upper left', frameon=True)
        # naming the x axis
        plt.xlabel('Age')
        # naming the y axis
        plt.ylabel('Correct Responses')

        # giving a title to my graph
        plt.title('ASD vs Non-ASD Correct Responses vs Age Eye Tracker Experiment')

        # function to show the plot
        plt.show()

        print("hello")