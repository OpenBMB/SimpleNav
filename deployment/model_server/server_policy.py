"""Load a saved model once and serve reset/predict/close sessions."""

import argparse
import logging

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-path")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--idle-timeout", type=int, default=-1)
    args = parser.parse_args()
    from starVLA.model.framework.base_framework import baseframework

    relocation = {}
    if args.backbone_path:
        relocation["base_vlm"] = args.backbone_path
    if args.attn_implementation:
        relocation["attn_implementation"] = args.attn_implementation
    model = baseframework.from_pretrained(
        args.checkpoint, config_overrides={"framework": {"qwenvl": relocation}} if relocation else None
    )
    model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    WebsocketPolicyServer(
        model, host=args.host, port=args.port, idle_timeout=args.idle_timeout, metadata={"checkpoint": args.checkpoint}
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
