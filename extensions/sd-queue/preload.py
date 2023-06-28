def preload(parser):
    parser.add_argument(
        '--start-task-listener',
        action='store_true',
        help='Enable pulsar message monitoring',
        default=None
    )