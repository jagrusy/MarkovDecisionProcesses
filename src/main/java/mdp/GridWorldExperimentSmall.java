package main.java.mdp;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.io.IOException;
import java.util.List;

public class GridWorldExperimentSmall {
    private final int width;
    private final int height;
    int[][] map;
    static double probTxSucceed = 0.8;
    GridWorldDomain gwdg;
    OOSADomain domain;
    TerminalFunction tf;
    RewardFunction rf;
    StateConditionTest goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;


    public GridWorldExperimentSmall() {
        this.width  = 11;
        this.height = 11;
        map1();
        gwdg = new GridWorldDomain(this.width, this.height);
        gwdg.setMap(this.map);
        gwdg.setProbSucceedTransitionDynamics(probTxSucceed);
        tf = new GridWorldTerminalFunction(0, this.height-1);
        gwdg.setTf(tf);
        goalCondition = new TFGoalCondition(tf);
        rf = new GoalBasedRF(this.goalCondition, 5.0, -0.5);
        gwdg.setRf(rf);
        domain = gwdg.generateDomain();

        initialState = new GridWorldState(new GridAgent(this.width-1, 0), new GridLocation(0, this.height-1, "loc0"));
        hashingFactory = new SimpleHashableStateFactory();

        env = new SimulatedEnvironment(domain, initialState);

        // Visualizes the agent running in real time
        // VisualActionObserver observer = new VisualActionObserver(domain,
        // 	GridWorldVisualizer.getVisualizer(gwdg.getMap()));
        // observer.initGUI();
        // env.addObservers(observer);
    }


    public void visualize(String outputpath){
        Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
        new EpisodeSequenceVisualizer(v, domain, outputpath);
    }

    public void valueIterationExample(String outputPath, double gamma){

        Planner planner = new ValueIteration(domain, gamma, hashingFactory, 0.001, 100);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "vi");

        simpleValueFunctionVis((ValueFunction)planner, p);

    }

    public void policyIterationExample(String outputPath, double gamma){

        Planner planner = new PolicyIteration(domain, gamma, hashingFactory, 0.001, 100, 100, 100);
        Policy p = planner.planFromState(initialState);

        PolicyUtils.rollout(p, initialState, domain.getModel()).write(outputPath + "pi");

        simpleValueFunctionVis((ValueFunction)planner, p);

    }

    public void qLearningExample(String outputPath, double gamma, double qInit, double learningRate, double epsilon, int numEpisodes){

        QLearning agent = new QLearning(domain, gamma, hashingFactory, qInit, learningRate);
        agent.setLearningPolicy(new EpsilonGreedy(agent, epsilon));
        //run learning for n episodes
        for(int i = 0; i < numEpisodes; i++){
            Episode e = agent.runLearningEpisode(env);

            e.write(outputPath + "ql_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }

    }

    public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p){

        List<State> allStates = StateReachability.getReachableStates(
                initialState, domain, hashingFactory);
        ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
                allStates, this.width, this.height, valueFunction, p);
        gui.initGUI();

    }

    public void experimentAndPlotter(double gamma, double qInit, double learningRate, double epsilon, int numEpisodes){

        //different reward function for more structured performance plots
        ((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Q-Learning";
            }

            public LearningAgent generateAgent() {
                QLearning agent = new QLearning(domain, gamma, hashingFactory, qInit, learningRate);
                agent.setLearningPolicy(new EpsilonGreedy(agent, epsilon));
                return agent;
            }
        };

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
                env, 10, numEpisodes, qLearningFactory);
        exp.setUpPlottingConfiguration(400, 400, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.STEPS_PER_EPISODE,
                PerformanceMetric.AVERAGE_EPISODE_REWARD);

        long start = System.currentTimeMillis();
        exp.startExperiment();
        System.out.println(System.currentTimeMillis()-start);
        String filepath = String.format("qlearning_%sdf_%sepsilon_%slr_", gamma, epsilon, learningRate);
        exp.writeStepAndEpisodeDataToCSV(filepath);

    }

    public void map1() {
        int[][] map =  {
                {0,0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,1,0},
                {1,1,1,1,1,1,0,1,1,1,0},
                {0,0,0,0,0,0,0,0,0,1,0},
                {0,0,0,0,0,0,0,0,0,1,0},
                {0,0,0,0,0,0,0,0,1,1,0},
                {1,1,1,1,0,1,1,1,1,0,0},
                {0,0,0,0,0,0,0,0,1,0,1},
                {0,0,0,0,0,0,0,0,1,0,0},
                {0,1,1,1,1,1,1,1,1,1,0},
                {0,0,0,0,0,0,0,0,0,0,0}
        };
        this.map = map;
    }

    public static void main(String[] args) throws IOException {
        GridWorldExperimentSmall gwe = new GridWorldExperimentSmall();
        String outputPath = "output/";

        String algorithm = "VI";
        double gamma = 0.9;
        double qInit = 0.3;
        double learningRate = 1.0;
        double epsilon = 0.1;
        int numEpisodes = 50;
        long start = System.currentTimeMillis();

        if (algorithm == "PI") {
            gwe.policyIterationExample(outputPath, gamma);
        } else if (algorithm == "VI") {
            gwe.valueIterationExample(outputPath, gamma);
        } else if (algorithm == "QL") {
            gwe.qLearningExample(outputPath, gamma, qInit, learningRate, epsilon, numEpisodes);
        }
        long wallClock = System.currentTimeMillis() - start;
        System.out.printf("Wall clock for %s: %d\n", algorithm, wallClock);
        if (algorithm == "QL") {
            gwe.experimentAndPlotter(gamma, qInit, learningRate, epsilon, numEpisodes);
            System.out.printf("%s_%sepsilon_%sLR_%d#eps_%dms", algorithm, epsilon, learningRate, numEpisodes, wallClock);
        } else {
            System.out.printf("%s_%sdf_%stxd_%dms", algorithm, gamma, probTxSucceed, wallClock);
        }
        gwe.visualize(outputPath);

    }

}
