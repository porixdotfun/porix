import base58
from solana.publickey import PublicKey
from solana.keypair import Keypair
from solana.transaction import Transaction
from solana.system_program import SYS_PROGRAM_ID, create_account
import solana.spl.token.instructions as spl_token
from solana.spl.token.constants import TOKEN_PROGRAM_ID

from blockchain.solana_client import SolanaClient
from utils.logging import get_logger
from config import CONFIG

logger = get_logger(__name__)


class TokenManager:
    """Manager for SPL token operations"""

    def __init__(self, solana_client=None):
        self.solana_client = solana_client or SolanaClient()
        self.token_mint = CONFIG.PORIX_TOKEN_ADDRESS

        # Initialize token mint if needed
        if not self.token_mint or self.token_mint == "":
            logger.info("No token mint address configured, creating new token")
            self.token_mint = self._create_token_mint()

        if isinstance(self.token_mint, str):
            self.token_mint = PublicKey(self.token_mint)

        logger.info(f"Token manager initialized with mint: {self.token_mint}")

    def _create_token_mint(self):
        """Create a new token mint"""
        try:
            # Generate keypair for token mint
            token_keypair = Keypair()

            # Get recent blockhash
            blockhash_resp = self.solana_client.client.get_recent_blockhash()
            blockhash = blockhash_resp["result"]["value"]["blockhash"]

            # Create transaction
            transaction = Transaction()
            transaction.recent_blockhash = blockhash

            # Calculate rent-exempt minimum balance
            rent_resp = self.solana_client.client.get_minimum_balance_for_rent_exemption(
                spl_token.MINT_LEN
            )
            mint_rent = rent_resp["result"]

            # Add instruction to create account for token mint
            transaction.add(
                create_account(
                    {
                        "from_pubkey": self.solana_client.keypair.public_key,
                        "to_pubkey": token_keypair.public_key,
                        "lamports": mint_rent,
                        "space": spl_token.MINT_LEN,
                        "program_id": TOKEN_PROGRAM_ID
                    }
                )
            )

            # Add instruction to initialize mint
            transaction.add(
                spl_token.initialize_mint(
                    token_keypair.public_key,
                    CONFIG.PORIX_TOKEN_DECIMALS,
                    self.solana_client.keypair.public_key,
                    None  # Freeze authority
                )
            )

            # Sign and send transaction
            transaction.sign(self.solana_client.keypair, token_keypair)

            resp = self.solana_client.client.send_transaction(
                transaction, self.solana_client.keypair, token_keypair
            )

            if "result" in resp:
                logger.info(f"Token mint created: {token_keypair.public_key}")
                return token_keypair.public_key
            else:
                logger.error(f"Failed to create token mint: {resp}")
                return None

        except Exception as e:
            logger.error(f"Error creating token mint: {str(e)}")
            return None

    def mint_tokens(self, to_pubkey, amount):
        """
        Mint tokens to a recipient

        Args:
            to_pubkey: Recipient's public key
            amount: Amount to mint

        Returns:
            str: Transaction signature
        """
        if isinstance(to_pubkey, str):
            to_pubkey = PublicKey(to_pubkey)

        try:
            # Get recent blockhash
            blockhash_resp = self.solana_client.client.get_recent_blockhash()
            blockhash = blockhash_resp["result"]["value"]["blockhash"]

            # Create transaction
            transaction = Transaction()
            transaction.recent_blockhash = blockhash

            # Find or create destination token account
            token_account = self.solana_client._find_or_create_token_account(
                self.token_mint, to_pubkey, True, transaction
            )

            if not token_account:
                logger.error("Failed to find or create token account")
                return None

            # Add mint instruction
            transaction.add(
                spl_token.mint_to(
                    self.token_mint,
                    token_account,
                    self.solana_client.keypair.public_key,
                    amount
                )
            )

            # Sign and send transaction
            transaction.sign(self.solana_client.keypair)

            resp = self.solana_client.client.send_transaction(
                transaction, self.solana_client.keypair
            )

            if "result" in resp:
                signature = resp["result"]
                logger.info(f"Tokens minted successfully: {signature}")
                return signature
            else:
                logger.error(f"Failed to mint tokens: {resp}")
                return None

        except Exception as e:
            logger.error(f"Error minting tokens: {str(e)}")
            return None

    def burn_tokens(self, from_pubkey, amount, from_token_account=None):
        """
        Burn tokens from an account

        Args:
            from_pubkey: Account owner's public key
            amount: Amount to burn
            from_token_account: Source token account (default: find owned account)

        Returns:
            str: Transaction signature
        """
        if isinstance(from_pubkey, str):
            from_pubkey = PublicKey(from_pubkey)

        try:
            # Get recent blockhash
            blockhash_resp = self.solana_client.client.get_recent_blockhash()
            blockhash = blockhash_resp["result"]["value"]["blockhash"]

            # Create transaction
            transaction = Transaction()
            transaction.recent_blockhash = blockhash

            # Find source token account if not provided
            if not from_token_account:
                token_accounts = self.solana_client.get_token_accounts(from_pubkey, self.token_mint)
                if not token_accounts:
                    logger.error(f"No token account found for {from_pubkey}")
                    return None
                from_token_account = PublicKey(token_accounts[0]["pubkey"])
            elif isinstance(from_token_account, str):
                from_token_account = PublicKey(from_token_account)

            # Add burn instruction
            transaction.add(
                spl_token.burn(
                    from_token_account,
                    self.token_mint,
                    from_pubkey,
                    amount
                )
            )

            # Sign and send transaction
            transaction.sign(self.solana_client.keypair)

            resp = self.solana_client.client.send_transaction(
                transaction, self.solana_client.keypair
            )

            if "result" in resp:
                signature = resp["result"]
                logger.info(f"Tokens burned successfully: {signature}")
                return signature
            else:
                logger.error(f"Failed to burn tokens: {resp}")
                return None

        except Exception as e:
            logger.error(f"Error burning tokens: {str(e)}")
            return None

    def get_token_supply(self):
        """
        Get the current token supply

        Returns:
            int: Current token supply
        """
        try:
            response = self.solana_client.client.get_token_supply(self.token_mint)
            if "result" in response and "value" in response["result"]:
                return int(response["result"]["value"]["amount"])
            else:
                logger.error(f"Unexpected token supply response: {response}")
                return 0
        except Exception as e:
            logger.error(f"Error getting token supply: {str(e)}")
            return 0

    def transfer_tokens(self, from_pubkey, to_pubkey, amount):
        """
        Transfer tokens between accounts

        Args:
            from_pubkey: Sender's public key
            to_pubkey: Recipient's public key
            amount: Amount to transfer

        Returns:
            str: Transaction signature
        """
        return self.solana_client.transfer_token(
            self.token_mint, to_pubkey, amount, None, True
        )

    def get_token_balance(self, public_key):
        """
        Get token balance for a user

        Args:
            public_key: User's public key

        Returns:
            int: Token balance
        """
        if isinstance(public_key, str):
            public_key = PublicKey(public_key)

        token_accounts = self.solana_client.get_token_accounts(public_key, self.token_mint)
        if not token_accounts:
            return 0

        token_account = PublicKey(token_accounts[0]["pubkey"])
        return self.solana_client.get_token_balance(token_account)

    def create_token_account(self, owner_pubkey):
        """
        Create a token account for an owner

        Args:
            owner_pubkey: Owner's public key

        Returns:
            PublicKey: Token account public key
        """
        if isinstance(owner_pubkey, str):
            owner_pubkey = PublicKey(owner_pubkey)

        try:
            # Get recent blockhash
            blockhash_resp = self.solana_client.client.get_recent_blockhash()
            blockhash = blockhash_resp["result"]["value"]["blockhash"]

            # Create transaction
            transaction = Transaction()
            transaction.recent_blockhash = blockhash

            # Create associated token account
            associated_token_address = self.solana_client._get_associated_token_address(
                self.token_mint, owner_pubkey
            )

            # Check if account exists
            account_info = self.solana_client.client.get_account_info(associated_token_address)
            if "result" in account_info and account_info["result"]["value"] is None:
                # Account doesn't exist, create it
                transaction.add(
                    spl_token.create_associated_token_account(
                        self.solana_client.keypair.public_key,
                        owner_pubkey,
                        self.token_mint
                    )
                )

                # Sign and send transaction
                transaction.sign(self.solana_client.keypair)

                resp = self.solana_client.client.send_transaction(
                    transaction, self.solana_client.keypair
                )

                if "result" not in resp:
                    logger.error(f"Failed to create token account: {resp}")
                    return None

            return associated_token_address

        except Exception as e:
            logger.error(f"Error creating token account: {str(e)}")
            return None