from deadly_corridor import (
    PPOTrainer,
    CURRICULUM_STAGES,
    PPO_CFG,
    CKPT_CFG,
    ENV_CFG,
    set_global_seeds,
)


def main():
    set_global_seeds()
    trainer = PPOTrainer(
        stages=CURRICULUM_STAGES,
        ppo_cfg=PPO_CFG,
        ckpt_cfg=CKPT_CFG,
        env_cfg=ENV_CFG,
    )
    trainer.train()


if __name__ == "__main__":
    main()
